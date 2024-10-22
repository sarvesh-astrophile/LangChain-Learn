from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
)

chat_history = []

system_message = SystemMessage(
    content="You are an AI customer support agent",
)
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    response = model.invoke(chat_history).content
    chat_history.append(AIMessage(content=response))
    print(f"AI: {response}")

print("--- Message History ---")
print(chat_history)
