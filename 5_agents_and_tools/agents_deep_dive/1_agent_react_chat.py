from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain import hub

# Load environment variables
load_dotenv()

# Define the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# Define the tools
def get_current_datetime(*args, **kwargs):
    """Get the current date and time"""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def search_wikipedia(query):
    """Search Wikipedia for a given query"""
    import wikipedia
    try:
        # If query is a dict, extract the 'value' key
        if isinstance(query, dict) and 'value' in query:
            query = query['value']
        # If query is a list, join all elements
        elif isinstance(query, list):
            query = ' '.join(map(str, query))
        # Convert to string if it's not already
        query = str(query)
        
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.PageError:
        return "No results found"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation error: {e.options}"

# List the tools available to the agent
tools = [Tool(name="get_current_datetime", func=get_current_datetime, description="Get the current date and time"),
         Tool(name="search_wikipedia", func=search_wikipedia, description="Search Wikipedia for when you need to know something about a topic")]

# Load the current JSON prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Create the agent from the prompt and LLM
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Create Buffer memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parse_errors=True)

# Initial system message to set the context for the chat
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
# Instead of adding a SystemMessage, add it as a HumanMessage
memory.chat_memory.add_message(HumanMessage(content=f"System: {initial_message}"))

# Chat with the agent
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    # add the user input to the memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    
    # invoke the agent and the current chat history
    response = agent_executor.invoke({"input": user_input})
    print("Agent:", response["output"])
    
    # add the agent response to the memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
    
    # print the chat history
    print("\n---Chat History Starts---")
    for message in memory.chat_memory.messages:
        print(f"{message.type}: {message.content}")
    print("---Chat History Ends---\n")
