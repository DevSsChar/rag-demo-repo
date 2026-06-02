# RAG (Retrieval Augmented Generation) - Comprehensive Guide

Welcome to the RAG project! This is a complete implementation guide for building Retrieval Augmented Generation systems using LangChain. This project covers the entire pipeline: from loading various data sources to intelligently chunking documents for optimal retrieval.

---

## Project Structure

```
d:\rag\
├── README.md                          # Main project documentation (you are here)
├── loaders/                           # Document loading implementations
│   ├── README.md                      # Detailed loaders guide
│   ├── csv_loader.py                  # CSV file loader
│   ├── pdf_loader.py                  # PDF document loader
│   ├── text_loader.py                 # Plain text file loader
│   ├── web_loader.py                  # Web content loader
│   ├── directory_loader.py            # Batch directory loader
│   ├── modified_placement_data.csv    # Sample CSV data
│   ├── Finance Dashboard UI.txt       # Sample text data
│   └── README.md                      # Loaders documentation
│
├── splitters/                         # Text splitting implementations
│   ├── README.md                      # Detailed splitters guide
│   ├── text_splitter.py               # Character-based splitter
│   ├── document-based/
│   │   └── document.py                # Document-based splitter
│   ├── length-based/
│   │   ├── text_splitter.py           # Length-based splitting
│   │   └── doc_splitter.py            # Document length splitter
│   ├── semantic-meaning-based/
│   │   └── semantic.py                # Semantic chunking with embeddings
│   └── structure-based/
│       └── structure_based_splitter.py # Structure-aware splitting
│
├── retrievers/                        # Retrieval implementations
│   └── langchain_retrievers.ipynb     # Various retrieval strategies (Wikipedia, Vector Store, MMR, MultiQuery, Contextual Compression)
│
└── rag1/                              # Python virtual environment
    └── ...
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (included as `rag1/`)
- Required API keys (OpenAI for semantic splitting)

### Activation

Activate the Python virtual environment:

**Windows (PowerShell):**
```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& d:\rag\rag1\Scripts\Activate.ps1)
```

**Windows (Command Prompt):**
```cmd
d:\rag\rag1\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source d:/rag/rag1/bin/activate
```

### Installation

Install required packages:

```bash
# Core dependencies
pip install langchain-community langchain-text-splitters python-dotenv

# For PDF processing
pip install pypdf

# For web scraping
pip install beautifulsoup4

# For semantic splitting (uses semantic-chunker-langchain instead of deprecated langchain-experimental)
pip install semantic-chunker-langchain langchain-openai

# For LLM integration
pip install langchain-groq
```

Or install all at once:

```bash
pip install langchain-community langchain-text-splitters python-dotenv pypdf beautifulsoup4 semantic-chunker-langchain langchain-openai langchain-groq
```

**Note:** The project now uses `semantic-chunker-langchain` for semantic splitting, as `langchain-experimental` is deprecated and no longer actively maintained.

### Environment Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-key-here
GROQ_API_KEY=your-groq-key-here
USER_AGENT=MyRAGApplication/1.0
```

**⚠️ Security Best Practice:** Never hardcode API keys in your scripts or notebooks. Always use environment variables:

```python
from dotenv import load_dotenv
import os

# Load from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
```

Add `.env` to your `.gitignore` to prevent accidentally committing secrets:

```
.env
```

---

## Component Guide

### 1. Document Loaders

Document loaders are the first step in your RAG pipeline. They transform various data sources into LangChain Document objects.

**📖 Full Documentation:** [loaders/README.md](loaders/README.md)

#### Supported Formats

| Format | File | Use Case |
|--------|------|----------|
| **Text** | [text_loader.py](loaders/text_loader.py) | Plain text files (.txt) |
| **PDF** | [pdf_loader.py](loaders/pdf_loader.py) | PDF documents |
| **CSV** | [csv_loader.py](loaders/csv_loader.py) | Tabular data |
| **Web** | [web_loader.py](loaders/web_loader.py) | URLs and web content |
| **Directory** | [directory_loader.py](loaders/directory_loader.py) | Batch load multiple files |

#### Quick Example

```python
# Text Loading
from langchain_community.document_loaders import TextLoader

loader = TextLoader(r'd:\rag\Finance Dashboard UI.txt', encoding='utf-8')
docs = loader.load()
print(f"Loaded {len(docs)} documents")
```

**→ [See all loader examples](loaders/README.md)**

---

### 2. Text Splitters

Text splitters break down large documents into manageable chunks for processing and retrieval. Different strategies suit different use cases.

**📖 Full Documentation:** [splitters/README.md](splitters/README.md)

#### Splitting Strategies

| Strategy | File | Best For | Speed |
|----------|------|----------|-------|
| **Character-based** | [splitters/text_splitter.py](splitters/text_splitter.py) | Quick tests | ⚡⚡⚡ |
| **Recursive** | [splitters/length-based/](splitters/length-based/) | General purpose | ⚡⚡ |
| **Document-based** | [splitters/document-based/document.py](splitters/document-based/document.py) | Preserve integrity | ⚡⚡⚡ |
| **Semantic** | [splitters/semantic-meaning-based/semantic.py](splitters/semantic-meaning-based/semantic.py) | High quality (uses semantic-chunker-langchain) | ⚡ |
| **Structure-based** | [splitters/structure-based/structure_based_splitter.py](splitters/structure-based/structure_based_splitter.py) | Markdown/HTML | ⚡⚡ |

#### Quick Example

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")
```

**→ [See all splitter examples](splitters/README.md)**

---

## Common Workflows

### Workflow 1: Load and Split a Text File

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load document
loader = TextLoader(r'd:\rag\Finance Dashboard UI.txt', encoding='utf-8')
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

print(f"Loaded: {len(docs)} documents, {len(chunks)} chunks created")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1} ({len(chunk.page_content)} chars):")
    print(chunk.page_content[:100] + "...")
```

### Workflow 2: Process Multiple PDF Files

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load all PDFs from directory
loader = DirectoryLoader(
    path=r'd:\rag\documents',
    glob="*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)
docs = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")
```

### Workflow 3: Semantic-Based Chunking

```python
from semantic_chunker_langchain.chunker import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# Load document
loader = TextLoader(r'd:\rag\Finance Dashboard UI.txt', encoding='utf-8')
docs = loader.load()

# Semantic splitting with embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

chunks = splitter.create_documents([docs[0].page_content])
print(f"Created {len(chunks)} semantic chunks")
```

### Workflow 4: Load CSV Data

```python
from langchain_community.document_loaders import CSVLoader

# Load CSV file
loader = CSVLoader(
    file_path=r'd:\rag\loaders\modified_placement_data.csv',
    encoding='utf-8'
)
docs = loader.load()

print(f"Loaded {len(docs)} rows from CSV")
for i, doc in enumerate(docs[:3]):
    print(f"\nRow {i+1}:")
    print(doc.page_content)
```

---

## Best Practices

### Security

✅ **Do:**
- Use `.env` files to manage API keys (load with `dotenv`)
- Never hardcode secrets in code or notebooks
- Add `.env` to `.gitignore`
- Use environment variables for all credentials
- Rotate API keys periodically

❌ **Don't:**
- Commit `.env` files to version control
- Hardcode API keys in scripts or notebooks
- Share code with exposed credentials
- Use the same API key across projects
- Log or print sensitive information

### Data Loading

✅ **Do:**
- Always specify encoding explicitly (`encoding='utf-8'`)
- Use raw string paths (`r'path'`) to avoid backslash issues
- Handle errors gracefully with try-except blocks
- Validate data after loading

❌ **Don't:**
- Load entire large files into memory at once
- Ignore encoding errors
- Skip metadata preservation
- Process without validation

### Document Splitting

✅ **Do:**
- Start with `RecursiveCharacterTextSplitter` for general use
- Use `chunk_overlap` to maintain context (20% of chunk_size)
- Test different chunk sizes on your data
- Monitor chunk statistics

❌ **Don't:**
- Use fixed character positions without semantic understanding
- Create chunks that are too small (< 100 chars)
- Ignore document boundaries and structure
- Skip testing with actual data

### RAG Pipeline

✅ **Do:**
- Combine multiple loaders for diverse data sources
- Use appropriate chunk sizes (typically 500-2000 chars)
- Preserve document metadata throughout pipeline
- Validate chunks before indexing

❌ **Don't:**
- Load all documents at once (use lazy loading)
- Mix different chunk sizes without reason
- Lose source information
- Skip quality assurance

---

## Troubleshooting

### Issue: ModuleNotFoundError for SemanticChunker

**Problem:** `ModuleNotFoundError: No module named 'langchain_experimental'` or `No module named 'semantic_chunker_langchain'`

**Solution:**
```bash
# Use the new semantic-chunker-langchain package
pip install semantic-chunker-langchain
```

The project has migrated from the deprecated `langchain-experimental` to `semantic-chunker-langchain` for semantic splitting.

### Issue: ModuleNotFoundError

**Problem:** `ModuleNotFoundError: No module named 'langchain_text_splitters'`

**Solution:**
```bash
pip install langchain-text-splitters
```

### Issue: PDF Text Extraction Issues

**Problem:** PDF text appears corrupted or incomplete

**Solution:**
- Verify PDF quality: `loader = PyPDFLoader(pdf_path)`
- Some PDFs require OCR (Optical Character Recognition)
- Use alternative loaders for complex PDFs

### Issue: Semantic Splitting Costs

**Problem:** API costs are too high for semantic splitting

**Solution:**
- Use `text-embedding-3-small` model (cheaper)
- Process documents in batches
- Implement caching for embeddings
- Use character-based splitting for testing

### Issue: Large Files Memory Issues

**Problem:** Loading large files causes out-of-memory errors

**Solution:**
```python
# Use lazy loading instead
docs_lazy = loader.lazy_load()

# Process one document at a time
for doc in docs_lazy:
    chunks = splitter.split_documents([doc])
    # Process chunks...
```

---

## Performance Tips

### Optimize Loading

```python
# Use lazy_load() for large file collections
docs = loader.lazy_load()

# Process incrementally
for doc in docs:
    # Process one document
    pass
```

### Optimize Splitting

```python
# For speed: Use RecursiveCharacterTextSplitter
# For quality: Use SemanticChunker
# For balance: RecursiveCharacterTextSplitter with larger overlap
```

### Monitor Performance

```python
import time

start = time.time()
docs = loader.load()
elapsed = time.time() - start
print(f"Loading time: {elapsed:.2f}s")

start = time.time()
chunks = splitter.split_documents(docs)
elapsed = time.time() - start
print(f"Splitting time: {elapsed:.2f}s")
print(f"Avg chunk time: {elapsed/len(chunks)*1000:.2f}ms")
```

---

## Sample Data

This project includes sample data files for testing:

- **Text:** [Finance Dashboard UI.txt](loaders/Finance%20Dashboard%20UI.txt)
- **CSV:** [modified_placement_data.csv](loaders/modified_placement_data.csv)

Use these files to test loaders and splitters without external dependencies.

---

## Next Steps

1. **Read the Loaders Guide:** [loaders/README.md](loaders/README.md)
   - Understand different data sources
   - Choose appropriate loaders
   - Learn best practices

2. **Read the Splitters Guide:** [splitters/README.md](splitters/README.md)
   - Explore splitting strategies
   - Understand trade-offs
   - Optimize for your use case

3. **Explore Retrievers:** [retrievers/langchain_retrievers.ipynb](retrievers/langchain_retrievers.ipynb)
   - Wikipedia retriever
   - Vector store retriever (Chroma, FAISS)
   - MMR (Maximum Marginal Relevance) search
   - MultiQuery retriever
   - Contextual compression retriever

4. **Build Your Pipeline:**
   - Combine loaders and splitters
   - Test with sample data
   - Iterate and optimize

5. **Integrate with Retrieval:**
   - Store chunks in vector database
   - Implement similarity search
   - Build complete RAG application

---

## Resources

### LangChain Documentation

- [Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)
- [Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Embeddings](https://python.langchain.com/docs/modules/data_connection/text_embedding/)
- [Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)

### RAG Best Practices

- [Building RAG Applications](https://python.langchain.com/docs/use_cases/question_answering/)
- [Advanced RAG Patterns](https://python.langchain.com/docs/use_cases/question_answering/chat_history/)
- [Production RAG Guide](https://python.langchain.com/docs/guides/deployments/)

### Vector Databases

- [Pinecone](https://www.pinecone.io/)
- [Chroma](https://www.trychroma.com/)
- [Weaviate](https://weaviate.io/)
- [Milvus](https://milvus.io/)

---

## File Directory Reference

Quick navigation to key files:

| Component | Location |
|-----------|----------|
| **Loaders Guide** | [loaders/README.md](loaders/README.md) |
| **Splitters Guide** | [splitters/README.md](splitters/README.md) |
| **Retrievers Notebook** | [retrievers/langchain_retrievers.ipynb](retrievers/langchain_retrievers.ipynb) |
| **Text Loader** | [loaders/text_loader.py](loaders/text_loader.py) |
| **PDF Loader** | [loaders/pdf_loader.py](loaders/pdf_loader.py) |
| **CSV Loader** | [loaders/csv_loader.py](loaders/csv_loader.py) |
| **Web Loader** | [loaders/web_loader.py](loaders/web_loader.py) |
| **Directory Loader** | [loaders/directory_loader.py](loaders/directory_loader.py) |
| **Character Splitter** | [splitters/text_splitter.py](splitters/text_splitter.py) |
| **Semantic Splitter** | [splitters/semantic-meaning-based/semantic.py](splitters/semantic-meaning-based/semantic.py) |
| **Structure Splitter** | [splitters/structure-based/structure_based_splitter.py](splitters/structure-based/structure_based_splitter.py) |

---

## Contributing

To add new loaders or splitters:

1. Create a new file in the appropriate directory
2. Document with docstrings and comments
3. Add to this README with examples
4. Test with sample data
5. Update the relevant component guide

---

## License

This project is for educational purposes.

---

## Support

For issues or questions:

1. Check the relevant component guide ([loaders/README.md](loaders/README.md) or [splitters/README.md](splitters/README.md))
2. Review the troubleshooting section
3. Test with sample data files included in the project

---

**Happy RAG building!** 🚀

Start with the loaders guide → choose your data sources → pick a splitting strategy → build your retrieval system!
