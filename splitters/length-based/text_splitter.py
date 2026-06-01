from langchain_text_splitters import CharacterTextSplitter
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

print(text)

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''  
)

# if docs then use splitter.split_documents(docs) else use splitter.split_text(text)
result=splitter.split_text(text)
print(result)