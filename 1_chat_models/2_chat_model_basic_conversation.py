from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
)

# messages = [
#     SystemMessage(content="Solve the math problems"),
#     HumanMessage(content="What is 81 divided by 9?"),
# ]

# result = model.invoke(messages)
# print("Answer from AI: ", result.content)

messages = [
    SystemMessage(content="Solve the math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content=" 81 divided by 9 is 9"),
    HumanMessage(content="What is 10 times 5")
]

result = model.invoke(messages)
print("Answer from AI: ", result.content)
