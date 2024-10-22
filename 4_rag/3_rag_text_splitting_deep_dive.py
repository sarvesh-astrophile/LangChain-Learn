import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter, TextSplitter, TokenTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List

# Define the directory containing the text files
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db", "chroma_db_random")

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Load the text file
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Function to creat and persist the vector store
def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(current_dir, "db", store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating new vector store {store_name} ---")
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"\n--- Vector store {store_name} created and persisted ---")
    else:
        print(f"\n--- Loading existing vector store {store_name} ---")
        db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)

# 1. Character based text splitter
print("\n--- Using Character based text splitter ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")

# 2. Sentence based text splitter
print("\n--- Using Sentence based text splitter ---")
sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sentence_docs = sentence_splitter.split_documents(documents)
create_vector_store(sentence_docs, "chroma_db_sentence")

# 3. Token based text splitter
print("\n--- Using Token based text splitter ---")
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")

# 4. Recursive character based text splitter
print("\n--- Using Recursive character based text splitter ---")
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
recursive_docs = recursive_splitter.split_documents(documents)
create_vector_store(recursive_docs, "chroma_db_recursive")

# 5. Custom text splitter (use paragraph as chunk)
print("\n--- Using Custom text splitter ---")
class CustomTextSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        return text.split("\n\n")

custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")

# function to query a vector store
def query_vector_store(query, store_name):
    persistent_directory = os.path.join(current_dir, "db", store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying vector store {store_name} ---")
        db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)
        retriver = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.1})
        relevant_docs = retriver.invoke(query)
        # print the relevant docs
        print(f"\n--- Relevant docs for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"\n--- Document {i} ---")
            print(doc.page_content)
            if doc.metadata:
                print(f"Metadata: {doc.metadata}")
    else:
        raise ValueError(f"The vector store {store_name} does not exist.")
    
# Define the query
query = "How did Juliet die?"

# query the vector stores
query_vector_store(query, "chroma_db_char")
query_vector_store(query, "chroma_db_sentence")
query_vector_store(query, "chroma_db_token")
query_vector_store(query, "chroma_db_recursive")
query_vector_store(query, "chroma_db_custom")
