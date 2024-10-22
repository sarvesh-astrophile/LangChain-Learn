import os
from typing import Type

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

# Load environment variables 
load_dotenv()

# Pydantic models for tool argumnets
class SimpleSearchInput(BaseModel):
    query: str = Field(description="The query to search the web with")

class MultiplyNumberArgs(BaseModel):
    a: float = Field(description="The first number to multiply")
    b: float = Field(description="The second number to multiply")

# Custom tool with only custom input
class SimpleSearchTool(BaseTool):
    name = "simple_search"
    description = "Search the web for information"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str) -> str:
        """Use the tool."""
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        response = client.search(query)
        return f"Search results for {query}: {response}"

# Custom tool with only custom output
class MultiplyNumberTool(BaseTool):
    name = "multiply_number"
    description = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyNumberArgs

    def _run(self, a: float, b: float) -> float:
        """Use the tool."""
        return a * b


# Create tools using the pydantic subclass approach
tools = [SimpleSearchTool(), MultiplyNumberTool()]

# get google api key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initalize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=google_api_key)

# Pull the prompt from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create an agent with the tools
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parse_errors=True)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Search for Apple Intelligence"})
print("Response for 'Search for LangChain updates':", response)

response = agent_executor.invoke({"input": "Multiply 10 and 20"})
print("Response for 'Multiply 10 and 20':", response)


