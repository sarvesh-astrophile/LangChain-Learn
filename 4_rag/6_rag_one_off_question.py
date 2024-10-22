import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables
load_dotenv()

# Define the current directory and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Define the vector store
db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)

# Define the query
query = "How did Juliet die?"

# Retrieve the relevant documents based on the query
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
relevant_docs = retriever.invoke(query)

# Display the retrieved results with metadata
print("\n--- Retrieved documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata['source']}")

# Combine the query and retrieved documents into a single string
combined_input = ("Here are some documents that might be relevant to your question. "
                  "Please answer the question based on the documents provided. "
                  "If you cannot answer the question based on the documents, "
                  "say so. Don't make up an answer.\n"
                  f"Query: {query}\n\nDocuments:\n")

document_contents = [f"Document {i}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs, 1)]
combined_input += "\n\n".join(document_contents)

# Define the ChatOpenAI instance
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the message for the model
messages = [SystemMessage(content="You are a helpful assistant that can answer questions based on the provided documents."),
            HumanMessage(content=combined_input)]

# Get the response from the model
print(messages)
result = model.invoke(messages)

# Dispaly the full result and content only
print("\n--- Generated answer ---")
print(result.content)
