from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()

# Functions for the tools
def greet_user(name: str) -> str:
    """Greet the user with a personalized message."""
    return f"Hello {name}!"

def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]

def get_current_date_time(*args, **kwargs) -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def concatenate_strings(a: str, b: str) -> str:
    """Concatenate two strings."""
    return a + b

# Pydantic models for tools arguments
class ConcatenateStringsArgs(BaseModel):
    a: str = Field(..., description="The first string to concatenate")
    b: str = Field(..., description="The second string to concatenate")

# Create the tools using the Tool and StructuredTool constructors
tools = [
    Tool(
        name="greet_user",
        description="Greet the user with a personalized message.",
        func=greet_user,
    ),
    Tool(
        name="get_current_date_time",
        description="Get the current date and time.",
        func=get_current_date_time,
    ),
    Tool(
        name="reverse_string",
        description="Reverse a string.",
        func=reverse_string,
    ),

    StructuredTool.from_function(
        name="concatenate_strings",
        description="Concatenate two strings",
        func=concatenate_strings,
        args_schema=ConcatenateStringsArgs,
    )
]

# print env variables
api_key = os.environ["GOOGLE_API_KEY"]
print(f"GOOGLE_API_KEY: {api_key}")

# Initialize the ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# Pull the prompt from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

#  Test the agent with sample queries
response = agent_executor.invoke({"input": "Greet Alice"})
print(response)

response = agent_executor.invoke({"input": "What is the current time ? format it as x:xx am/pm"})
print(response)

response = agent_executor.invoke({"input": "Reverse the string 'Hello, world!'"})
print(response)

response = agent_executor.invoke({"input": "Concatenate 'Hello'and 'Sarvesh'"})
print(response)


