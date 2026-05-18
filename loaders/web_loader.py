# pip install bs4 since uses BeautifulSoup under the hood to parse the web page
# import WebBaseLoader from langchain_community.document_loaders
import json
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# for loading environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Prevent USER_AGENT warning
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "MyRAGApplication/1.0"

# Add friendly error if GROQ_API_KEY is missing
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file: GROQ_API_KEY=your_groq_api_key")

# put the url of the web page you want to load here
url="https://en.wikipedia.org/wiki/Artificial_intelligence"
loader=WebBaseLoader(url)

# load the documents from the web page
docs=loader.load()

# print the loaded documents
# print(docs[0].page_content)
# print(docs[0].metadata)

# summarize via groq
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