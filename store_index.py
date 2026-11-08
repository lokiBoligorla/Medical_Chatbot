
from dotenv import load_dotenv
import os

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Validate API key
if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY not found. Check your .env file."
    )


# Step 1: Extract PDF data
print("Loading PDF files...")

extracted_data = load_pdf_file(data="data/")

if not extracted_data:
    raise ValueError("No PDF documents found inside data/ folder.")


# Step 2: Filter documents
print("Filtering documents...")

filter_data = filter_to_minimal_docs(extracted_data)


# Step 3: Split documents into chunks
print("Splitting documents into chunks...")

text_chunks = text_split(filter_data)

print(f"Total text chunks: {len(text_chunks)}")


# Step 4: Load HuggingFace embeddings
print("Loading HuggingFace embeddings...")

embeddings = download_hugging_face_embeddings()


# Step 5: Initialize Pinecone
print("Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"


# Step 6: Create Pinecone index if it doesn't exist
if not pc.has_index(index_name):

    print("Creating Pinecone index...")

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

else:
    print("Pinecone index already exists.")


# Step 7: Store documents and embeddings
print("Uploading documents to Pinecone...")

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)


print("Successfully stored documents in Pinecone!")
