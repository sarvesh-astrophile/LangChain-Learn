from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes.")
])

# Create a RunnableSequence chain
chain = prompt_template | model | StrOutputParser()

# Run the chain
result = chain.invoke({"topic": "chickens", "joke_count": 2})
print(result)

