from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are an helpful assistant."),
    ("human", "Generate a thankyou note for this positive feedback: {feedback}")
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are an helpful assistant."),
    ("human", "Generate a response addressing this negative feedback: {feedback}")
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are an helpful assistant."),
    ("human", "Generate a request for more details for this neutral feedback: {feedback}")
])

escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are an helpful assistant."),
    ("human", "Generate a message escalating this feedback to a human agent: {feedback}")
])

# Define the feedback classification template
feedback_classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are an helpful assistant."),
    ("human", "Classify this feedback as positive, negative, neutral, or escalate a manager: {feedback}")
])

# Create Runnables for each feedback type
positive_feedback_chain = positive_feedback_template | model | StrOutputParser()
negative_feedback_chain = negative_feedback_template | model | StrOutputParser()
neutral_feedback_chain = neutral_feedback_template | model | StrOutputParser()
escalate_feedback_chain = escalate_feedback_template | model | StrOutputParser()

class PrintBranchCallback(BaseCallbackHandler):
    def __init__(self, branch_name):
        self.branch_name = branch_name

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Branch: {self.branch_name}")

# Define the runnable branches for handling different feedback types
branches = RunnableBranch(
    (lambda x: "positive" in x, positive_feedback_chain.with_config({"callbacks": [PrintBranchCallback("Positive")]})),
    (lambda x: "negative" in x, negative_feedback_chain.with_config({"callbacks": [PrintBranchCallback("Negative")]})),
    (lambda x: "neutral" in x, neutral_feedback_chain.with_config({"callbacks": [PrintBranchCallback("Neutral")]})),
    escalate_feedback_chain.with_config({"callbacks": [PrintBranchCallback("Escalate")]})
)

# Create the classification chain
classification_chain = feedback_classification_template | model | StrOutputParser()

# Combine the classification and response generation into one chain
chain = classification_chain | branches

# Examples reviews
positive_review = "I love the new iPhone 15! The camera is amazing and the battery life is great."
negative_review = "I am very disappointed with the new iPhone 15. The camera is terrible and the battery life is short."
neutral_review = "The new iPhone 15 is good. The camera is average and the battery life is okay."
escalate_review = "This is a serious issue with the new iPhone 15. I want to speak to a manager immediately."

# Example feedback
feedback = neutral_review

# Run the feedback through the chain
result = chain.invoke({"feedback": feedback})

print("Result:", result)
