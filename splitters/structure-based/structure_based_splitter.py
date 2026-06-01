from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader=TextLoader(r'D:\rag\splitters\Finance Dashboard UI.txt',encoding='utf-8')

# load the documents from the text file
docs=loader.load()

# print the loaded documents
# print(docs)

text=""
for doc in docs:
    text+=doc.page_content
    # print("content:", doc.page_content)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

chunks=splitter.split_text(text)
print(len(chunks))

