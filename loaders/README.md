# LangChain Document Loaders - Comprehensive Guide

This guide explains the intuition, use cases, limitations, and implementation details of various document loaders in LangChain. Document loaders are the first step in any RAG (Retrieval Augmented Generation) pipeline, transforming various data sources into LangChain `Document` objects.

---

## Installation Requirements

Before using these loaders, install the required packages:

```bash
# Core dependencies
pip install langchain-community
pip install python-dotenv

# For PDF loading
pip install pypdf

# For Web loading
pip install beautifulsoup4

# For LLM integration (optional, used in some examples)
pip install langchain-groq

# For advanced loaders
pip install selenium              # For JavaScript-rendered content
pip install unstructured          # For complex document types
```

Or install all at once:

```bash
pip install langchain-community pypdf beautifulsoup4 python-dotenv langchain-groq
```

---

## Table of Contents

1. [Text Loader](#text-loader)
2. [PDF Loader](#pdf-loader)
3. [CSV Loader](#csv-loader)
4. [Web Loader](#web-loader)
5. [Directory Loader](#directory-loader)
6. [Custom Loader](#custom-loader)
7. [Explore More Loaders](#explore-more-loaders)

---

## Text Loader

### Intuition

The `TextLoader` is the simplest loader designed for plain text files (.txt). It reads the entire file content and converts it into a single LangChain `Document` object with metadata (file path, source, etc.). Think of it as the foundational building block for loading unstructured text data.

### Use Cases

- Loading configuration files, logs, or documentation
- Processing plain text documents
- Reading markdown files or transcripts
- Quick prototyping with simple text sources

### Limitations

- Entire file is treated as one document; no automatic chunking
- No intelligent parsing; formatting is preserved as-is
- Memory inefficient for very large files
- Encoding issues may occur with non-UTF-8 files
- No support for multi-document batching

### Code Snippet

```python
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# Create a TextLoader instance
loader = TextLoader(r'path/to/your/file.txt', encoding='utf-8')

# Load documents (returns a list of Document objects)
docs = loader.load()

# Access document content and metadata
print(docs[0].page_content)  # The actual text content
print(docs[0].metadata)      # {'source': 'path/to/your/file.txt'}
```

### Best Practices

- Always specify encoding explicitly to avoid encoding errors
- For large files, consider splitting the document after loading
- Use raw string prefix (`r''`) to avoid issues with backslashes in file paths

---

## PDF Loader

### Intuition

The `PyPDFLoader` extracts text from PDF files, creating one `Document` object per page. PDFs are complex formats combining text, images, and formatting. PyPDF handles the extraction by parsing the PDF structure and converting it into readable text.

### Use Cases

- Loading research papers, reports, or white papers
- Extracting text from generated invoices or receipts
- Processing academic documents or textbooks
- Building searchable indexes of PDF archives

### Limitations

- Extracts only text; images and complex layouts are lost
- Quality depends on PDF structure; some PDFs have encoding issues
- Metadata extraction may be incomplete
- Performance degrades with very large PDFs (100+ pages)
- Special characters or non-ASCII text may be corrupted
- Cannot reliably extract structured data from tables

### Code Snippet

```python
from langchain_community.document_loaders import PyPDFLoader

# Create a PyPDFLoader instance
loader = PyPDFLoader(r'path/to/your/document.pdf')

# Load documents (one per page)
docs = loader.load()

# Each document represents a page
print(f"Total pages: {len(docs)}")
print(f"Page 1 content: {docs[0].page_content}")
print(f"Page 1 metadata: {docs[0].metadata}")  # {'source': '...', 'page': 0}
```

### Best Practices

- Check PDF quality before processing; some PDFs may have encoding issues
- Use page metadata to track document structure
- Consider splitting large PDFs into chunks after extraction
- For tables, manually extract or use OCR-based loaders

---

## CSV Loader

### Intuition

The `CSVLoader` reads CSV files and converts each row into a `Document` object with the row data. It treats each row as a separate document, making it ideal for tabular data that needs to be queried individually.

### Use Cases

- Loading structured data from spreadsheets or databases
- Processing survey responses or customer data
- Querying product catalogs or inventories
- Analyzing sales data or records
- RAG systems over tabular business data

### Limitations

- Large CSV files can result in many documents, increasing processing overhead
- Does not preserve row relationships (each row is independent)
- No automatic aggregation or grouping
- Poor performance with extremely wide tables (many columns)
- Column headers become part of every document, increasing size
- No built-in data type handling; all values are treated as strings

### Code Snippet

```python
from langchain_community.document_loaders import CSVLoader

# Create a CSVLoader instance
loader = CSVLoader(
    file_path=r'path/to/data.csv',
    encoding='utf-8'
)

# Load documents (one per row)
docs = loader.load()

# Each document represents one row with all columns
print(f"Total rows: {len(docs)}")
print(f"First row: {docs[0].page_content}")
print(f"Metadata: {docs[0].metadata}")
```

### Best Practices

- Ensure CSV headers are descriptive for context
- Consider filtering rows before loading if the file is very large
- Use custom column selection if only specific fields are needed
- Combine with document splitting for better chunking strategies

---

## Web Loader

### Intuition

The `WebBaseLoader` fetches HTML content from web pages and extracts readable text. It uses BeautifulSoup under the hood to parse HTML and strip out code/markup, leaving only the human-readable content.

### Use Cases

- Loading web articles or blog posts
- Scraping documentation websites
- Extracting content from Wikipedia or similar sources
- Building knowledge bases from public web content
- Real-time content updates from live websites

### Limitations

- JavaScript-rendered content is not loaded (requires Selenium for that)
- Website structure changes may break parsing
- Rate limiting and bot detection challenges
- Cannot extract data behind paywalls or login requirements
- Large websites may timeout during loading
- Metadata extraction is limited
- May include unwanted navigation or footer content
- Requires USER_AGENT header to avoid being blocked

### Code Snippet

```python
import os
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

# Set USER_AGENT to identify your requests
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "MyRAGApplication/1.0"

# Create a WebBaseLoader instance
loader = WebBaseLoader(url="https://en.wikipedia.org/wiki/Artificial_intelligence")

# Load documents
docs = loader.load()

# Access loaded content
print(f"Page content length: {len(docs[0].page_content)}")
print(f"Metadata: {docs[0].metadata}")  # {'source': 'URL', 'title': 'Page Title'}
```

### Best Practices

- Always set a USER_AGENT to avoid being blocked
- Use error handling for network issues
- Consider implementing retry logic for flaky connections
- For JavaScript-heavy sites, use SeleniumURLLoader instead
- Respect robots.txt and website terms of service

---

## Directory Loader

### Intuition

The `DirectoryLoader` recursively loads multiple files from a directory using a specified loader class. Instead of loading one file at a time, it automates batch loading with pattern matching (globbing) to select specific file types.

### Use Cases

- Loading all documents from a folder of PDFs
- Batch processing multiple text files
- Processing document collections (research papers, reports)
- Building comprehensive knowledge bases from file systems
- Handling complex directory structures with multiple file types

### Limitations

- All documents must be the same type (requires separate loaders for mixed types)
- Globbing patterns can be complex and error-prone
- Progress tracking overhead can slow down loading with very large directories
- No support for nested directory filtering (glob selects all matching files recursively)
- Memory usage grows with the number of documents
- Performance degrades exponentially with thousands of files

### Code Snippet

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Create a DirectoryLoader instance
loader = DirectoryLoader(
    path=r'D:\rag\pdfs',           # Directory path
    glob="*.pdf",                   # Pattern to match PDF files
    show_progress=True,             # Display loading progress
    loader_cls=PyPDFLoader          # Loader class to use
)

# Load all documents
docs = loader.load()

# For large document collections, use lazy loading
# Documents are loaded one by one instead of all at once
docs_lazy = loader.lazy_load()

print(f"Total documents loaded: {len(docs)}")
print(f"First document metadata: {docs[0].metadata}")

# Iterate over lazy-loaded documents without loading all into memory
for doc in docs_lazy:
    print(f"Processing: {doc.metadata['source']}")
    # Do something with each document
```

### Best Practices

- Use `lazy_load()` for large collections to reduce memory usage
- Set `show_progress=True` for visibility during long operations
- Use specific glob patterns to avoid loading unwanted files
- Combine with document splitting for better handling of large files

---

## Custom Loader

### Intuition

A custom loader allows you to implement your own document loading logic. Extend the `BaseLoader` class to handle specialized file formats, APIs, or data sources that LangChain doesn't support out of the box.

### Use Cases

- Loading proprietary file formats
- Fetching data from custom APIs or databases
- Implementing specialized parsing logic
- Integrating with internal data systems
- Handling complex data transformations

### Implementation Example

```python
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from typing import List
import json

class JSONLinesLoader(BaseLoader):
    """
    Custom loader for JSON Lines format (one JSON object per line).
    
    JSON Lines is a convenient format for storing or streaming structured data
    that may be processed one record at a time.
    
    Example usage:
        loader = JSONLinesLoader("data.jsonl")
        docs = loader.load()
    """
    
    def __init__(self, file_path: str, metadata_fields: List[str] = None):
        """
        Initialize the JSON Lines loader.
        
        Args:
            file_path: Path to the JSONL file
            metadata_fields: List of field names to extract as metadata
        """
        self.file_path = file_path
        self.metadata_fields = metadata_fields or []
    
    def load(self) -> List[Document]:
        """Load and parse the JSON Lines file."""
        documents = []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    try:
                        # Parse JSON object from line
                        data = json.loads(line)
                        
                        # Extract metadata
                        metadata = {
                            'source': self.file_path,
                            'line': line_num
                        }
                        
                        # Add custom metadata fields
                        for field in self.metadata_fields:
                            if field in data:
                                metadata[field] = data[field]
                        
                        # Create document from JSON object
                        content = json.dumps(data, indent=2)
                        doc = Document(
                            page_content=content,
                            metadata=metadata
                        )
                        documents.append(doc)
                    
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
        
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        return documents
    
    def lazy_load(self):
        """Lazy load documents one by one (generator)."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        metadata = {
                            'source': self.file_path,
                            'line': line_num
                        }
                        
                        for field in self.metadata_fields:
                            if field in data:
                                metadata[field] = data[field]
                        
                        content = json.dumps(data, indent=2)
                        yield Document(
                            page_content=content,
                            metadata=metadata
                        )
                    
                    except json.JSONDecodeError:
                        continue
        
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")


# Usage example
if __name__ == "__main__":
    # Create sample JSONL file for testing
    sample_data = [
        {"id": 1, "name": "Alice", "role": "Engineer"},
        {"id": 2, "name": "Bob", "role": "Manager"},
        {"id": 3, "name": "Charlie", "role": "Designer"}
    ]
    
    with open("data.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    # Load documents using custom loader
    loader = JSONLinesLoader(
        "data.jsonl",
        metadata_fields=["id", "name"]
    )
    
    # Method 1: Load all at once
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    print(f"First document:\n{docs[0]}")
    
    # Method 2: Lazy load
    print("\nLazy loading:")
    for doc in loader.lazy_load():
        print(f"Loaded: {doc.metadata}")
```

### Key Points for Custom Loaders

- Inherit from `BaseLoader` class
- Implement `load()` method for batch loading
- Implement `lazy_load()` method for memory-efficient streaming
- Always include source metadata
- Handle errors gracefully
- Use type hints for clarity
- Support encoding specification for text-based formats

---

## Explore More Loaders

LangChain supports dozens of document loaders. Here are resources to discover and learn about more:

### Official LangChain Documentation

- **Main Loader Documentation**: https://python.langchain.com/docs/integrations/document_loaders/
- **Loader API Reference**: https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.base.BaseLoader.html

### Commonly Used Alternative Loaders

- **SeleniumURLLoader**: For JavaScript-rendered web content (requires Selenium)
- **GitLoader**: For loading files from Git repositories
- **JSONLoader**: For parsing JSON files with JSON pointers for content extraction
- **MarkdownHeaderSplitter**: For parsing markdown with hierarchical structure
- **UnstructuredLoader**: For complex document types (images, scanned PDFs, Office files)
- **NotionDBLoader**: For loading Notion databases
- **SlackDirectoryLoader**: For loading Slack channel history
- **ArxivLoader**: For loading academic papers from ArXiv
- **YouTubeLoader**: For loading YouTube video transcripts

### Installation for Advanced Loaders

```bash
# For web scraping with JavaScript
pip install selenium

# For unstructured documents
pip install unstructured

# For specific integrations
pip install langchain-community  # Contains most loaders

# For API-based sources
pip install langchain-openai langchain-anthropic
```

---

## Common Patterns

### Loading and Splitting

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load document
loader = TextLoader("document.txt", encoding='utf-8')
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks from {len(docs)} documents")
```

### Error Handling

```python
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def safe_load_pdf(pdf_path: str):
    """Load PDF with error handling."""
    try:
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return docs
    
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []
```

### Batch Loading Multiple Sources

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader
)

def load_multiple_sources(text_files, pdf_files, csv_files):
    """Load documents from multiple sources."""
    all_docs = []
    
    # Load text files
    for txt_file in text_files:
        try:
            loader = TextLoader(txt_file, encoding='utf-8')
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
    
    # Load PDF files
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    # Load CSV files
    for csv_file in csv_files:
        try:
            loader = CSVLoader(csv_file, encoding='utf-8')
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return all_docs
```

---

## Summary Table

| Loader | Format | Use Case | Best For | Limitations |
|--------|--------|----------|----------|------------|
| TextLoader | .txt | Simple text files | Quick prototyping | No chunking, memory inefficient |
| PyPDFLoader | .pdf | PDF documents | Research papers | Text extraction only, no images |
| CSVLoader | .csv | Tabular data | Business data, records | Row independence, wide tables |
| WebBaseLoader | URL | Web content | Articles, documentation | JS content, rate limiting |
| DirectoryLoader | Multiple | Batch processing | Document collections | Same file type required |
| Custom | Any | Specialized formats | APIs, proprietary formats | Requires implementation |

---

## Next Steps

1. Start with the simple `TextLoader` for basic use cases
2. Explore the relevant loader for your data source
3. Implement error handling and validation
4. Consider document splitting after loading
5. For complex scenarios, build a custom loader
6. Check the official documentation for the latest loaders

Happy loading!
