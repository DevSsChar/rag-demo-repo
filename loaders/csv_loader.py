from langchain_community.document_loaders import CSVLoader

loader=CSVLoader(file_path=r'D:\rag\modified_placement_data.csv', encoding='utf-8')

docs=loader.load()

print(docs[0])