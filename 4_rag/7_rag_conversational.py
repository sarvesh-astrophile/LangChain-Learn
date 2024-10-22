import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Define the current directory and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Define the vector store
db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)


# Create a retriever for querying the vector store
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create a chatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_question_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference content in the chat history, "
    "reformulate the standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return the question unchanged."
)

# Create a prompt template for the contextualize question chain
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_question_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on the chat history
history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=contextualize_question_prompt
)

# Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer the question."
    "If you don't know the answer, just say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "Do NOT make up an answer. \n\n"
    "{context}"
)

# Create a prompt template for the answer question
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create a chain to combine documents for question answering
# ``create_stuff_documents_chain`` is a simple chain that combines the documents
# retrieved from the vector store and passes them to the LLM
question_answering_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retriever chain that combines the history-aware retriever with the
# question answering chain
retriever_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)


# Function to simulate a continual chat
def chat_with_ai():
    chat_history = []
    print("Welcome to the AI assistant! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        result = retriever_chain.invoke({"input": user_input, "chat_history": chat_history})
        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(result["answer"])

# Main function to start the chat
if __name__ == "__main__":
    chat_with_ai()
