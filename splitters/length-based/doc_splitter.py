from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# put the path to your PDF file here, r for raw format to avoid issues with backslashes in the file path
loader=PyPDFLoader(r'D:\rag\splitters\Rag Task 0 to 1.pdf')

# load the documents from the PDF file
docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

results=splitter.split_documents(docs)

print(results[0].page_content)