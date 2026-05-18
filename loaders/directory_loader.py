# import directory loader and pdf loader
from langchain.document_loaders import DirectoryLoader,PyPDFLoader

loader=DirectoryLoader(
    path=r'D:\rag\pdfs', # path to the dir 
    glob="*.pdf", # to specify what files we want to load, here pdf files
    show_progress=True, # to show progress while loading documents
    loader_cls=PyPDFLoader # to provide the loader class to load the docs
    )

# load the documents from the directory
docs=loader.load()

# however, if we have a lot of docs, then above method can be slow, so instead we can use lazy loading, which loads the documents one by one when we access them, instead of loading all at once
docs=loader.lazy_load()

print(len(docs))

# to check the content of the first document
print(docs[0].page_content)
# to check the metadata of the first document
print(docs[0].metadata)

# similalrly for rest
print(docs[12].page_content)
print(docs[12].metadata)