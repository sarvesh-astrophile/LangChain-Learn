import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the path to the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding function
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embeddings function
db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)

# Define the user's query
query = "Who is Odysseus' wife?"

# Search the vector store for the query
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4, "k": 3})
relevant_docs = retriever.invoke(query)

# Display the relevant documents with metadata
for i, doc in enumerate(relevant_docs, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")

