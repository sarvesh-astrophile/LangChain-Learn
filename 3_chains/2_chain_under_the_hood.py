from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes.")
])

# Create individual Runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Combine them into a RunnableSequence chain
chain = RunnableSequence(
    first=format_prompt,
    middle=[invoke_model],
    last=parse_output
)

# Run the chain
result = chain.invoke({"topic": "chickens", "joke_count": 2})
print(result)


