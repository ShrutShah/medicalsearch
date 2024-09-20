# Project Report: Medical Research Hybrid Search

## Introduction

The objective of this project is to create a hybrid search system that can efficiently retrieve relevant medical information from a variety of sources, including web pages and PDF documents. The focus is on screening guidelines and preventive measures for cancers, particularly breast cancer and ovarian cancer.

## Project Overview

The project involves several key steps:

1. **Setting Up Pinecone**: A vector database to store embeddings for efficient retrieval.
2. **Data Extraction**: Scraping content from web pages and downloading PDFs.
3. **Data Loading**: Utilizing LangChain loaders to parse and manage documents.
4. **Text Chunking**: Splitting documents into manageable pieces for embedding.
5. **Creating Embeddings**: Using Hugging Face models to generate vector representations of text.
6. **BM25 Sparse Encoding**: Applying BM25 for text ranking and retrieval.
7. **Hybrid Search Implementation**: Integrating dense and sparse search capabilities.
8. **Querying the Index**: Testing the system with medical queries to evaluate performance.

## Step-by-Step Explanation

### 1. Setting Up Pinecone

Pinecone is a managed vector database that allows for efficient similarity search. The setup involves:

- **API Key**: A unique key to authenticate and connect to the Pinecone service.
- **Index Creation**: If the specified index does not exist, it is created with appropriate dimensions and metrics.

### 2. Data Extraction

Data is extracted from URLs using the `requests` library and `BeautifulSoup` for HTML parsing. The extraction process involves:

- **HTML Extraction**: Identifying headings (`h1`, `h2`, `h3`) and their corresponding content. The content is gathered until the next heading.
- **PDF Downloading**: If the URL points to a PDF, the file is downloaded and saved locally.

**Example URLs**:
- Cancer screening guidelines from the American Cancer Society.
- PDF documents from ICMR related to ovarian cancer.

### 3. Data Loading

The extracted HTML and PDF files are loaded into LangChain loaders:

- **PyPDFLoader**: For loading PDF documents.
- **UnstructuredHTMLLoader**: For loading HTML content.

This step transforms the raw content into structured documents, which can be easily processed.

### 4. Text Chunking

Documents are split into smaller text chunks to facilitate embedding:

- **Recursive Character Text Splitter**: Divides text into chunks of specified sizes while maintaining context.
- **Sentence Transformers Token Text Splitter**: Further splits the chunks into smaller pieces suitable for embedding.

### 5. Creating Embeddings

Using the `HuggingFaceEmbeddings` class, each chunk of text is converted into a dense vector representation. The model employed (`all-MiniLM-L6-v2`) is efficient and suitable for semantic search tasks.

### 6. BM25 Sparse Encoding

The BM25 algorithm is used to rank documents based on keyword matching. This involves:

- **Fitting the BM25 Encoder**: The encoder is trained on the final data chunks to learn the document distribution.
- **Dumping Encoder Values**: The learned BM25 values are saved for later use.

### 7. Hybrid Search Implementation

A hybrid retriever is set up to combine the strengths of both dense (embeddings) and sparse (BM25) retrieval methods. This enables:

- **Efficient Search**: The system can quickly find relevant documents based on semantic meaning and keyword relevance.

### 8. Querying the Index

The system is tested with a set of sample queries related to breast and ovarian cancer. Example queries include:

- **Preventive measures for breast cancer**.
- **Screening guidelines for ovarian cancer**.
- **Effectiveness of mammograms**.

Each query is processed through the retriever, and the results are returned based on relevance.

## Results

The hybrid search system successfully retrieved relevant documents based on the medical queries. The results demonstrated the system's ability to integrate different retrieval techniques effectively, providing users with comprehensive and pertinent information.

### Sample Query Results

1. **Query**: What are the preventive measures for breast cancer?
   - **Results**: Information about self-exams, mammograms, and lifestyle changes.

2. **Query**: What is a mammogram?
   - **Results**: Definitions and explanations about mammograms, including their role in early detection.

## Conclusion

This project successfully establishes a hybrid search system capable of efficiently retrieving relevant medical information from diverse sources. The integration of Pinecone, LangChain, and embedding techniques allows for a powerful search capability, addressing the needs of healthcare professionals and researchers seeking timely and accurate medical information.

## Future Work

Future enhancements could include:

- Expanding the dataset with more diverse sources and document types.
- Implementing a user-friendly interface for easier interaction with the search system.
- Exploring more advanced embedding techniques for improved retrieval accuracy.
