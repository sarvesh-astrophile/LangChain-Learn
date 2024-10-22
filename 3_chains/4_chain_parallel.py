from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnableParallel

load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert product reviewer."),
    ("human", "List the main features of {product_name}.")
])

# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert product reviewer."),
        ("human", "Given these features: {features}, list the pros of these features.")
    ])
    return pros_template.format_prompt(features=features)

# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert product reviewer."),
        ("human", "Given these features: {features}, list the cons of these features.")
    ])
    return cons_template.format_prompt(features=features)

# Combine pros and cons into a final review
def combine_review(pros, cons):
    return f"Pros: {pros}\nCons: {cons}"


# Simplify braches with LCEL
pro_branch_chain = RunnableLambda(analyze_pros) | model | StrOutputParser()
cons_branch_chain = RunnableLambda(analyze_cons) | model | StrOutputParser()

# Combine pros and cons into a final review
chain = prompt_template | model | StrOutputParser() | RunnableParallel(branches={"pros": pro_branch_chain, "cons": cons_branch_chain}) | RunnableLambda(lambda x: combine_review(x['branches']['pros'], x['branches']['cons']))

# Run the chain with an input
result = chain.invoke({"product_name": "iPhone 13"})
print(result)
