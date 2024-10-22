import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Define the persist directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(current_dir, "db", "chroma_db_apple")

# STEP 1: Scrape the content from apple.com using WebBaseLoader
urls = ["https://www.apple.com/"]

# Load the documents
loader = WebBaseLoader(urls)
documents = loader.load()

# STEP 2: Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of documents chunks: {len(docs)}")
print(f"Sample chunk: {docs[0].page_content}\n")

# STEP 3: Create a vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# STEP 4: Create and persist the vector store with the documents and embeddings
if not os.path.exists(persist_directory):   
    print(f"\n--- Creating vector store and persisting to {persist_directory} ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    print(f"--- Vector store created and persisted successfully ---")
else:
    print(f"\n--- Loading vector store from {persist_directory} ---")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print(f"--- Vector store loaded successfully ---")

# STEP 5: Query the vector store
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the user's question
user_question = "What new products are announced on Apple.com?"

# Retrieve relevant documents based on the user's question
relevant_docs = retriever.invoke(user_question)

# Display the relevant documents
print("\n--- Relevant Documents ---")
for i,doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n {doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata['source']}\n")