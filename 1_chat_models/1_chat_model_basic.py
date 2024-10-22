from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
)

# Invoke the model
result = model.invoke("What is 81 divided by 9?")
print("Full response: ", result)
print("Content: ", result.content)
