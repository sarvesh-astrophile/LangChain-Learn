from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


# # PART 1: Create a ChartPromptTemplate using a template string
# template = "Tell me a joke about {topic}"
# prompt_template = ChatPromptTemplate.from_template(template)

# print("---Prompt from Template---")
# prompt = prompt_template.invoke({"topic": "chickens"})
# print(prompt)

# # PART 2: Create a ChartPromptTemplate with Multiple Placeholders
# template_multiple = """ You are a helpful AI assistant.
# Human: Tell me a {adjective} story about a {animal}
# Assistant: """
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)

# print("---Prompt with Multiple Placeholders---")
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "elephant"})
# print(prompt)


# PART 3: Prompt with System and Human Message using Tuples
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)

print("---Prompt with System and Human Message using Tuples---")
prompt = prompt_template.invoke({"topic": "chickens", "joke_count": 2})
print(prompt)
