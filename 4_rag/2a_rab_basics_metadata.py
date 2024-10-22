import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Define the path to the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initilizing new vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"Books directory does not exist: {books_dir}")
    
    # List all text files in the books directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    for file in book_files:
        file_path = os.path.join(books_dir, file)
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": file}
            documents.append(doc)
    
    # Print first document and metadata
    print("\n--- First document ---")
    print(documents[0])
    print(documents[0].metadata)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document information ---")
    print(f"Number of documents: {len(documents)}")
    print(f"Number of chunks: {len(chunks)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("\n--- Finished creating embeddings ---")

    # Create the vector store
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("\n--- Vector store already exists ---")

    
