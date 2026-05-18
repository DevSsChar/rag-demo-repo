# pip install pypdf as pypdfloader uses pypdf under the hood
# load pypdfloader from langchain_community.document_loaders
from langchain_community.document_loaders import PyPDFLoader

# put the path to your PDF file here, r for raw format to avoid issues with backslashes in the file path
loader=PyPDFLoader(r'D:\rag\Rag Task 0 to 1.pdf')

# load the documents from the PDF file
docs=loader.load()

# print the loaded documents
print(docs[0])
print(docs[1].metadata)