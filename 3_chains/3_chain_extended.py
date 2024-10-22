from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda

load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes.")
])

# Create individual Runnables (steps in the chain)
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f'word count: {len(x.split())}\n{x}')

# Combine them into a RunnableSequence chain
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# Run the chain
result = chain.invoke({"topic": "chickens", "joke_count": 2})
print(result)
