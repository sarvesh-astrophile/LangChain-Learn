import os
from google.oauth2 import service_account
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""


load_dotenv()

# Path to your service account JSON key file
SERVICE_ACCOUNT_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

#setup firebase Firestore
PROJECT_ID = "langchain-chat-history-bafcb"
SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat_history"

# Initialize FireStore Client with service account credentials
print("Initializing Firestore Client...")
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY_PATH)
client = firestore.Client(project=PROJECT_ID, credentials=credentials)

# Initialize ChatMessageHistory
print("Initializing ChatMessageHistory...")
chat_history = FirestoreChatMessageHistory(
    collection=COLLECTION_NAME,
    session_id=SESSION_ID,
    client=client,
)

print("Chat History Initialized")
print("Current Chat History: ", chat_history.messages)

# Initialize OpenAI Model
print("Initializing OpenAI Model...")
model = ChatOpenAI(temperature=0)

# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

print("Start chatting with the AI. Type 'exit' to end the chat.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Add the user's message to the chat history
    chat_history.add_user_message(user_input)

    # Create the list of messages for the model
    messages = prompt.format_messages(
        history=chat_history.messages,
        input=user_input
    )

    # Get the model's response
    response = model.invoke(messages)

    # Add the AI's response to the chat history
    chat_history.add_ai_message(response.content)

    print("AI:", response.content)

print("Chat ended.")
