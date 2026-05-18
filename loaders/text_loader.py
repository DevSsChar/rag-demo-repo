# pip install langchain-community for TextLoader
# fetch Textloader from langchain_community.document_loaders
from langchain_community.document_loaders import TextLoader

# pip install langchain-groq for ChatGroq
# we are now working with langchain-groq for the Groq API, thus following dependencies are needed
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

# for loading environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# create an instance of TextLoader with the path to your text file and the encoding
# r here for raw format to avoid issues with backslashes in the file path
loader=TextLoader(r'D:\rag\Finance Dashboard UI.txt',encoding='utf-8')

# load the documents from the text file
docs=loader.load()

# print the loaded documents
print(docs)
# type(docs) is "list" of Document objects, where each Document object has a page_content attribute that contains the text content of the document

# for getting first doc get the first element of the list
print(docs[0])
# to get respective metadata, same for page_content
print(docs[0].metadata)

# https://console.groq.com/docs/quickstart ref for code below
# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

# Define the expected JSON structure
parser = JsonOutputParser(pydantic_object={
    "type": "object",
    "properties": {
        "summary": {"type": "string"}
    }
})

# Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """Provide a concise summary of the text into JSON with this structure:
        {{
            "summary": "summary text here"
        }}"""),
    ("user", "{input}")
])

# Create the chain that guarantees JSON output
chain = prompt | llm | parser

def summarize_document(text: str) -> dict:
    result = chain.invoke({"input": text})
    print(json.dumps(result, indent=2))

        
# Example usage
description = docs[0].page_content  

summarize_document(description)