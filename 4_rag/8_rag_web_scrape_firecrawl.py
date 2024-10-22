import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader

# Load environment variables
load_dotenv()


# Define the persist directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(current_dir, "db", "chroma_db_firecrawl")

# Define the `create_vector_store` function
def create_vector_store():
    """
    Crawl the website, split the documents into chunks, create embeddings, and store them in a vector database.
    """
    # Define the Firecrwal api key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY is not set in the environment variables.")
    
    print(f"API Key found: {'*' * (len(api_key) - 4)}{api_key[-4:]}")  # Print last 4 characters of API key
    
    # STEP 1: Crawl the website using FireCrawlLoader
    print("\n--- Crawling the website ---")
    url = "https://apple.com"
    try:
        loader = FireCrawlLoader(url=url, api_key=api_key, mode="scrape")
        documents = loader.load()
        print("\n--- Website crawled successfully ---")
    except Exception as e:
        print(f"\nError during web crawling: {str(e)}")
        print("Please check your FireCrawl API key and ensure it's valid and active.")
        return

    # Convert metadata values to strings if they are lists
    for doc in documents:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # STEP 2: Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents and a sample chunk
    print("\n--- Document Chunks Information ---")
    print(f"Number of documents: {len(documents)}")
    print(f"Sample chunk: {docs[0].page_content}")

    # STEP 3: Create embeddings and store in a vector database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # STEP 4: Create the vector database with the embeddings
    print("\n--- Creating the vector database ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    print("\n--- Vector database created successfully ---")



# Check if the chroma vector store already exists
if os.path.exists(persist_directory):
    print(f"\n--- The vector database already exists in {persist_directory} ---")
else:
    print(f"\n--- The vector database does not exist in {persist_directory} ---")
    create_vector_store()

# Load the vector database with the embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# STEP 5: Query the vector database
def query_vector_db(query):
    """
    Query the vector database and return the results.
    """
    # Create a retriever for querying the vector database
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Retrieve the documents based on the query
    relevant_docs = retriever.get_relevant_documents(query)

    # Print the relevant result with metadata
    print("\n--- Relevant results ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"\n--- Document {i} ---")
        print(doc.page_content)
        if doc.metadata:
            print(doc.metadata)

# Test the query function
query = "Apple Intelligence?"
query_vector_db(query)



