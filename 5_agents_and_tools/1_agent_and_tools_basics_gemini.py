from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain import hub

# Load environment variables
load_dotenv()

# Define a vary simple tool function that return the current date and time
def get_current_datetime(*args, **kwargs):
    """Get the current date and time"""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# List the tools available to the agent
tools = [Tool(name="get_current_datetime", func=get_current_datetime, description="Get the current date and time")]

# Pull the prompt from the hub
prompt = hub.pull("hwchase17/react")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt, stop_sequence=True)

# Create the agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Run the agent with a test query
response = agent_executor.invoke({"input": "What is the time formatted as x:xx pm/am?"})

# Print the output
print("Response:", response)
