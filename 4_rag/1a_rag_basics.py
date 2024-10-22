import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Define the path to the directory containing the text files and the persistent director
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initilizing new vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found at {file_path}")
    
    # Read the text content from the file
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Documnets Chunks Information ---")
    print(f"Number of documents: {len(docs)}")
    print(f"Sample document chunk: {docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Persistent directory already exists. Loading existing vector store...")

