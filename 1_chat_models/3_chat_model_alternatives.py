from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# setup environment variables and messages
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problem"),
    HumanMessage(content="What is 2 + 2?"),
]


# LangChain OpenAi Chat Model
openai_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
)

response = openai_model.invoke(messages)
print(f"OpenAI: {response.content}")

# LangChain Anthropic Chat Model
anthropic_model = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0.5,
)

response = anthropic_model.invoke(messages)
print(f"Anthropic: {response.content}")

# LangChain Google Generative AI Chat Model
google_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
)

response = google_model.invoke(messages)
print(f"Google Gemini Flash: {response.content}")
