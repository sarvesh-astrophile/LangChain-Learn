import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_openai")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with embeddings function
db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)

# Function to query the vector store with different search types and parameters
def query_vector_store(store_name, query, embedding_function=embeddings, search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.1}):
    persistent_directory = os.path.join(current_dir, "db", store_name)
    if os.path.exists(persistent_directory):
        db = Chroma(embedding_function=embedding_function, persist_directory=persistent_directory)
        
        retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        relevent_docs = retriever.invoke(query)

        # Display the retrieved results with metadata
        print("\n--- Retrieved documents ---")
        for i, doc in enumerate(relevent_docs, 1):
            print(f"\nDocument {i}\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata['source']}\n")
    else:
        print(f"Vector store with name {store_name} does not exist.")

# Define the user's query
query = "How did Juliet die?"

# Showcase different retrieval methods

# 1. Similarity Search
print("\n--- Using Similarity Search ---")
query_vector_store("chroma_db_openai", query, embeddings, search_type="similarity", search_kwargs={"k": 3})

# 2. Max Marginal Relevance Search (MMR)
print("\n--- Using Max Marginal Relevance Search (MMR) ---")
query_vector_store("chroma_db_openai", query, embeddings, search_type="mmr", search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5})

# 3. Similarity Score Threshold
print("\n--- Using Similarity Score Threshold ---")
query_vector_store("chroma_db_openai", query, embeddings, search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.1})

print("querying demostration with different search types completed.")



