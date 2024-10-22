import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Load the existing chroma vector store from the persisted directory
current_directory = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_directory, "..", "..", "4_rag", "db")
persist_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding function
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Check if the Chroma vector store already exists
if os.path.exists(persist_directory):
    print("---Loading existing vector store from", persist_directory, "---")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    raise FileNotFoundError(f"---The directory {persist_directory} does not exist---")

# Create a retriever for querying the vector store
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create a Gemini 1.5 Pro LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_question_system_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "reformulate the question to be a standalone question which can be understood without the chat history. "
    "Do NOT answer the question, only reformulate it if needed and otherwise just return it as is. "
)

# Create the prompt template for contextualize question
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_question_system_prompt),
    ("human", "{input}"),
])

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_question_prompt)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicate what to do if the answer is unknown
answer_question_system_prompt = (
    "You are an assitance for question answering tasks."
    "Use the following pieces of retrieved context to answer the question."
    "If you don't know the answer, just say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "\n\n{context}"
)

# Create a prompt template for answer question
answer_question_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_question_system_prompt),
    ("human", "{input}"),
])


# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` is used to feed all retrieved documents to the LLM
question_answering_chain = create_stuff_documents_chain(llm, answer_question_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)

# Setup the ReAct agent with douments store retriever
# Load the ReAct agent from the hub
react_docstore_prompt = hub.pull("hwchase17/react")

# Setup the tools (chat history) for the agent
tools = [Tool( 
    name="Answer questions",
    func = lambda input, **kwargs: rag_chain.invoke({"input": input, "chat_history": kwargs.get("chat_history", [])}),
    description="useful for when you need to answer questions about the context",
)]

# Create the ReAct agent with documents store retriever
agent = create_react_agent(llm, tools, prompt=react_docstore_prompt)

# Setup the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parse_errors=True)

# Initialize the chat history
chat_history = []

# Start the chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
    print(response["output"])
    chat_history.append(response["output"])

    # Update the chat history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response["output"]))
