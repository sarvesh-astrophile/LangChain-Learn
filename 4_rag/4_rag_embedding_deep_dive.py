import os
from langchain.embeddings import VoyageEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Define the directory contaning the text file and the persistance directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_path = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist")

# Read the text content from the file
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks information ---")
print(f"Number of chunks: {len(docs)}")
print(f"First chunk: {docs[0].page_content}")

# Function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    persist_directory = os.path.join(db_path, store_name)
    if not os.path.exists(persist_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        print(f"\n--- Vector store {store_name} created successfully ---")
    else:
        print(f"\n--- Vector store {store_name} already exists ---")

# 1. OpenAI Embeddings
print("\n--- OpenAI Embeddings ---")
openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
create_vector_store(docs, openai_embeddings, "chroma_db_openai")

# 2. Voyage Embeddings
print("\n--- Voyage Embeddings ---")
voyage_embeddings = VoyageEmbeddings(model="voyage-3-lite")
create_vector_store(docs, voyage_embeddings, "chroma_db_voyage")

print("\n--- Embedding demonstration form OpenAI and Voyage completed ---")

# Function to query a vector store
def query_vector_store(query, store_name, embeddings_function):
    persist_directory = os.path.join(db_path, store_name)
    if os.path.exists(persist_directory):
        print(f"\n--- Quering vector store {store_name} ---")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings_function)
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.5})
        # Display the retrieved chunks
        relevant_chunks = retriever.invoke(query)
        print(f"\n --- Retrieved documents for {store_name} ---")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"\n Document {i}:")
            print(f"  Content: {chunk.page_content}")
            if chunk.metadata:
                print(f"  Metadata: {chunk.metadata}")
    else:
        raise ValueError(f"The vector store {store_name} does not exist")
    

# Define the query
query = "Who is Odysseus' wife?"

# Query the vector stores
query_vector_store(query, "chroma_db_openai", openai_embeddings)
query_vector_store(query, "chroma_db_voyage", voyage_embeddings)

print("\n--- Querying the vector stores completed ---")
