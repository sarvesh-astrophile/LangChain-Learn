import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with embeddings function
db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)

# Define the user's query
query = "How did Juliet die?"

# Retriver relevent documents based on the user's query
retriver = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.1})
retriver_docs = retriver.invoke(query)

# Display the retrieved results with metadata
print("\n--- Retrieved documents ---")
for i, doc in enumerate(retriver_docs):
    print(f"\nDocument {i}\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}")

