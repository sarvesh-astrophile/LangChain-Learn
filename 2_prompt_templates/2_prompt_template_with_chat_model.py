from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# PART 1: Create a ChatPromptTemplate using a template string
print("---Prompt Template---")
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "chickens"})
result = model.invoke(prompt)
print(result.content)

# PART 2: Prompt with multiple Placeholders
print("---Prompt with Multiple Placeholders---")
template_multiple = """ You are a helpful AI assistant.
Human: Tell me a {adjective} short story about a {animal}
Assistant: """
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)

prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "elephant"})
print(prompt)
result = model.invoke(prompt)
print(result.content)

