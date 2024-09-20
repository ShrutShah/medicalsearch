import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import glob

# Step 1: Setup Pinecone
api_key = "your_pinecone_api_key"
index_name = "medical-research-hybrid-search"

pc = Pinecone(api_key=api_key)

# Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dense vector dimension
        metric='dotproduct',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)

# Step 2: Data Extraction Function
def extract_headings_and_content(url, file_number, output_folder="data"):
    try:
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        response = requests.get(url)
        if response.status_code == 200 and url.lower().endswith('.html'):
            if not os.path.exists(output_folder + '/html_data'):
                os.makedirs(output_folder + '/html_data')
            soup = BeautifulSoup(response.content, "html.parser")

            extracted_data = []
            for heading_tag in soup.find_all(['h1', 'h2', 'h3']):
                heading = heading_tag.text.strip()
                content = ""
                next_element = heading_tag.find_next_sibling()
                while next_element and next_element.name and not next_element.name.startswith('h'):
                    content += str(next_element)
                    next_element = next_element.find_next_sibling()

                extracted_data.append({'heading': heading, 'content': content})

            html_content = f"<html><body><h1>Extracted Data</h1><ul>"
            for item in extracted_data:
                html_content += f"<li><strong>{item['heading']}</strong>: {item['content']}</li>"
            html_content += "</ul></body></html>"

            output_file = os.path.join(output_folder + '/html_data', f"html_file_{file_number}.html")
            with open(output_file, "w", encoding='utf-8') as file:
                file.write(html_content)

            return f"HTML extraction successful for {url}, saved as {output_file}"

        elif response.status_code == 200 and url.lower().endswith('.pdf'):
            if not os.path.exists(output_folder + '/pdf_data'):
                os.makedirs(output_folder + '/pdf_data')
            output_file = os.path.join(output_folder + '/pdf_data', f"pdf_file_{file_number}.pdf")
            with open(output_file, 'wb') as file_:
                file_.write(response.content)
            return f"PDF download successful for {url}, saved as {output_file}"

        else:
            return f"Failed to fetch {url}: {response.status_code}"

    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

# Step 3: Extract documents
all_url = [
    'https://www.cancer.org/cancer/types/breast-cancer/screening-tests-and-early-detection.html',
    'https://www.cancer.org/cancer/types/breast-cancer/screening-tests-and-early-detection/american-cancer-society-recommendations-for-the-early-detection-of-breast-cancer.html',
    'https://www.cancer.org/cancer/types/breast-cancer/screening-tests-and-early-detection/mammograms.html',
    'https://www.cancer.org/cancer/types/breast-cancer/screening-tests-and-early-detection/breast-ultrasound.html',
    'https://www.cancer.org/cancer/types/breast-cancer/screening-tests-and-early-detection/breast-mri-scans.html',
    'https://www.cancer.org/cancer/types/breast-cancer/screening-tests-and-early-detection/breast-cancer-signs-and-symptoms.html',
    'https://main.icmr.nic.in/sites/default/files/guidelines/Ovarian_Cancer.pdf'
]

# Extract content from each URL
for i, url in enumerate(all_url, start=1):
    print(extract_headings_and_content(url, i))

# Step 4: Load extracted data into LangChain loaders
html_files = glob.glob(f"data/html_data/*.html")
pdf_files = glob.glob(f"data/pdf_data/*.pdf")

all_documents = [PyPDFLoader(file).load() if file.endswith(".pdf") else UnstructuredHTMLLoader(file).load() for file in pdf_files + html_files]

# Step 5: Split text into smaller chunks for embedding
splitter_1 = RecursiveCharacterTextSplitter(separators=['\n\n', '.', ' '], chunk_size=1000, chunk_overlap=50)
splitter_2 = SentenceTransformersTokenTextSplitter(tokens_per_chunk=256, chunk_overlap=10)

all_text_data = [doc.page_content.strip() for doc_set in all_documents for doc in doc_set if doc.page_content]

prefinal_data = splitter_1.split_text('\n'.join(all_text_data))
final_data = []
[final_data.extend(splitter_2.split_text(text)) for text in prefinal_data]

# Step 6: Prepare embeddings and sparse encoder (BM25)
os.environ["HF_TOKEN"] = "your_hf_token"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

bm25_encoder = BM25Encoder().default()
bm25_encoder.fit(final_data)
bm25_encoder.dump('bm25_values.json')

bm25_encoder = BM25Encoder().load('bm25_values.json')

# Step 7: Add data to Pinecone for hybrid search
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
retriever.add_texts(final_data)

# Step 8: Query the index with medical queries
sample_queries = [
    "What are the preventive measures for breast cancer?",
    "What are the screening guidelines for ovarian cancer?",
    "How effective are mammograms in breast cancer detection?",
    "What is the treatment for early-stage cervical cancer?"
]

for query in sample_queries:
    results = retriever.invoke(query)
    print(f"Results for '{query}':\n{results}\n")

# Additional queries
additional_queries = [
    'What represents about 30% of all cancers of the female genital organs?',
    'What are the major recommendations for staging?',
    'What is a mammogram?',
    'What are the symptoms of breast cancer?'
]

for query in additional_queries:
    results = retriever.invoke(query)
    print(f"Results for '{query}':\n{results}\n")

