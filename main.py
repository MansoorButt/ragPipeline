# =========================
# FastAPI Backend for PDF QA System (Azure Blob + Table Storage + Pinecone + Groq)
# =========================

# pip install fastapi uvicorn python-multipart azure-storage-blob azure-data-tables langchain langchain-community langchain-huggingface langchain-pinecone langchain-groq pinecone-client pypdf python-dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timezone
import io

from dotenv import load_dotenv

load_dotenv()

from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient, TableEntity

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

from pinecone import Pinecone, ServerlessSpec

# =========================
# CONFIG
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Azure Blob
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "pdf-documents")

# Azure Table Storage
TABLE_NAME = "pdfmetadata"

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legalaivector"

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# =========================
# INIT
# =========================

app = FastAPI(title="PDF QA Backend", version="2.0.0")

# CORS - Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Blob Client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# Create container if it doesn't exist
try:
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER)
    container_client.get_container_properties()
    print(f"✅ Using existing container: {AZURE_CONTAINER}")
except Exception:
    container_client = blob_service_client.create_container(AZURE_CONTAINER)
    print(f"✅ Created new container: {AZURE_CONTAINER}")

# Azure Table Storage Client
table_service_client = TableServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# Create table if it doesn't exist
try:
    table_client = table_service_client.create_table_if_not_exists(TABLE_NAME)
    print(f"✅ Using Azure Table: {TABLE_NAME}")
except Exception as e:
    print(f"⚠️ Table setup: {e}")
    table_client = table_service_client.get_table_client(TABLE_NAME)

# Pinecone - SDK v3+ syntax
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
try:
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Change to your region if needed
            )
        )
        print(f"✅ Created Pinecone index: {INDEX_NAME}")
    else:
        print(f"✅ Using existing Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"⚠️ Pinecone setup: {e}")

# Get index reference
index = pc.Index(INDEX_NAME)


# =========================
# MODELS
# =========================

class QuestionRequest(BaseModel):
    file_id: str
    question: str


class FileMetadata(BaseModel):
    file_id: str
    original_filename: str
    upload_date: str
    status: str
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None


# =========================
# HELPERS
# =========================

def save_metadata_to_table(file_id: str, original_filename: str, file_size: int, status: str = "uploaded"):
    """Save file metadata to Azure Table Storage"""
    try:
        entity = {
            "PartitionKey": "pdf_files",  # Group all PDFs together
            "RowKey": file_id,
            "original_filename": original_filename,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "file_size": file_size,
            "page_count": 0,
            "chunk_count": 0
        }
        table_client.upsert_entity(entity)
        print(f"✅ Saved metadata for {file_id}")
    except Exception as e:
        print(f"❌ Error saving metadata: {e}")


def update_metadata_status(file_id: str, status: str, page_count: int = 0, chunk_count: int = 0):
    """Update file processing status in Azure Table Storage"""
    try:
        entity = table_client.get_entity(partition_key="pdf_files", row_key=file_id)
        entity["status"] = status
        if page_count > 0:
            entity["page_count"] = page_count
        if chunk_count > 0:
            entity["chunk_count"] = chunk_count
        entity["last_updated"] = datetime.now(timezone.utc).isoformat()
        table_client.update_entity(entity, mode="replace")
        print(f"✅ Updated status for {file_id} to {status}")
    except Exception as e:
        print(f"❌ Error updating metadata: {e}")


def get_metadata_from_table(file_id: str) -> Optional[dict]:
    """Retrieve file metadata from Azure Table Storage"""
    try:
        entity = table_client.get_entity(partition_key="pdf_files", row_key=file_id)
        return dict(entity)
    except Exception as e:
        print(f"❌ Error fetching metadata: {e}")
        return None


def list_all_files_from_table() -> List[dict]:
    """List all files from Azure Table Storage"""
    try:
        entities = table_client.query_entities(query_filter="PartitionKey eq 'pdf_files'")
        return [dict(entity) for entity in entities]
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []


def process_pdf_from_blob(file_id: str):
    """Background task to process PDF from Azure Blob"""
    try:
        print(f"🔄 Starting processing for file_id: {file_id}")
        update_metadata_status(file_id, "processing")

        temp_path = os.path.join(TEMP_DIR, f"{file_id}.pdf")

        # Download from Azure Blob
        blob_client = container_client.get_blob_client(f"{file_id}.pdf")
        with open(temp_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        print(f"✅ Downloaded PDF to {temp_path}")

        # Load and split PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        page_count = len(documents)
        print(f"✅ Loaded {page_count} pages from PDF")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)
        chunk_count = len(docs)
        print(f"✅ Split into {chunk_count} chunks")

        # Create embeddings
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HF_API_KEY")
        )

        # Store in Pinecone
        from langchain_pinecone import PineconeVectorStore

        vectorstore = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=file_id
        )
        print(f"✅ Stored {chunk_count} vectors in Pinecone namespace: {file_id}")

        # Update metadata with success
        update_metadata_status(file_id, "ready", page_count, chunk_count)
        print(f"✅ Processing complete for file_id: {file_id}")

        # Cleanup
        os.remove(temp_path)

    except Exception as e:
        error_msg = f"error: {str(e)}"
        update_metadata_status(file_id, error_msg)
        print(f"❌ Error processing {file_id}: {str(e)}")


def validate_legal_document_with_llm(file_content: bytes) -> tuple[bool, str]:
    """
    Validate if PDF is a legal document using LLM classification.
    Analyzes first 2 pages to determine if it's a judgment, case law, legal study, etc.

    Returns: (is_valid, message)
    """
    try:
        # Save temporarily
        temp_path = os.path.join(TEMP_DIR, "temp_validate.pdf")
        with open(temp_path, "wb") as f:
            f.write(file_content)

        # Extract text
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        if not documents:
            return False, "PDF is empty or unreadable"

        # Get first 2 pages for validation (faster & cheaper)
        text = "\n".join([doc.page_content for doc in documents[:2]])

        if not text.strip():
            return False, "No text content found in PDF"

        # Clean up temp file
        os.remove(temp_path)

        # Use LLM for classification
        llm = ChatGroq(
            temperature=0,
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant"
        )

        prompt = ChatPromptTemplate.from_template("""
You are a legal document classifier. Analyze the following text and determine if it is a legal document.

Legal documents include:
- Court judgments and verdicts
- Case law and precedents  
- Legal opinions
- Statutes and legislation
- Regulations
- Contracts (legal agreements)
- Court orders
- Legal briefs and petitions
- Historical legal cases
- Legal studies and analyses

Respond with ONLY "YES" or "NO" followed by a brief reason (one line).

Text:
{text}
""")

        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"text": text[:2000]})  # Use first 2000 chars

        result_lower = result.lower().strip()

        if "yes" in result_lower:
            return True, "✅ Valid legal document confirmed by LLM"
        else:
            # Extract reason from LLM response
            reason = result.split('\n')[1] if '\n' in result else "Document does not appear to be legal"
            return False, f"❌ Not a legal document - {reason}"

    except Exception as e:
        return False, f"❌ Validation error: {str(e)}"

# =========================
# ROUTES
# =========================

@app.post("/upload", response_model=FileMetadata)
async def upload_pdf(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload PDF to Azure Blob and start background processing
    Returns file metadata including file_id for future reference
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_id = str(uuid.uuid4())
    original_filename = file.filename

    print(f"📤 Uploading file: {original_filename} with ID: {file_id}")

    # Read file content
    file_content = await file.read()
    file_size = len(file_content)

    # VALIDATION: Check if it's actually a legal document using LLM
    print(f"🔍 Validating if this is a legal document...")
    is_valid_legal_doc, validation_message = validate_legal_document_with_llm(file_content)

    if not is_valid_legal_doc:
        print(f"❌ Validation failed: {validation_message}")
        raise HTTPException(
            status_code=400,
            detail=f"Document validation failed: {validation_message}"
        )

    print(f"✅ {validation_message}")

    # Upload to Azure Blob
    blob_client = container_client.get_blob_client(f"{file_id}.pdf")
    blob_client.upload_blob(file_content, overwrite=True)
    print(f"✅ Uploaded to Azure Blob: {file_id}.pdf")

    # Save metadata to Table Storage
    save_metadata_to_table(file_id, original_filename, file_size, "uploaded")

    # Start background processing
    background_tasks.add_task(process_pdf_from_blob, file_id)
    print(f"🚀 Background task queued for {file_id}")

    return FileMetadata(
        file_id=file_id,
        original_filename=original_filename,
        upload_date=datetime.now(timezone.utc).isoformat(),
        status="uploaded",
        file_size=file_size
    )


@app.get("/files", response_model=List[FileMetadata])
def list_all_files():
    """
    Get all uploaded files with their metadata
    Perfect for displaying in a grid view on the frontend
    """
    files = list_all_files_from_table()

    return [
        FileMetadata(
            file_id=f["RowKey"],
            original_filename=f.get("original_filename", "Unknown"),
            upload_date=f.get("upload_date", ""),
            status=f.get("status", "unknown"),
            file_size=f.get("file_size"),
            page_count=f.get("page_count"),
            chunk_count=f.get("chunk_count")
        )
        for f in files
    ]


@app.get("/files/{file_id}", response_model=FileMetadata)
def get_file_metadata(file_id: str):
    """
    Get metadata for a specific file
    """
    metadata = get_metadata_from_table(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    return FileMetadata(
        file_id=metadata["RowKey"],
        original_filename=metadata.get("original_filename", "Unknown"),
        upload_date=metadata.get("upload_date", ""),
        status=metadata.get("status", "unknown"),
        file_size=metadata.get("file_size"),
        page_count=metadata.get("page_count"),
        chunk_count=metadata.get("chunk_count")
    )


@app.get("/files/{file_id}/download")
async def download_pdf(file_id: str):
    """
    Download/stream the actual PDF file
    Perfect for displaying in an iframe or PDF viewer on the frontend
    """
    # Check if file exists in metadata
    metadata = get_metadata_from_table(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Get the PDF from Azure Blob
        blob_client = container_client.get_blob_client(f"{file_id}.pdf")
        pdf_data = blob_client.download_blob().readall()

        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(pdf_data),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{metadata.get("original_filename", "document.pdf")}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@app.get("/status/{file_id}")
def check_status(file_id: str):
    """
    Check processing status of a file (legacy endpoint, use /files/{file_id} instead)
    """
    metadata = get_metadata_from_table(file_id)

    if not metadata:
        return {
            "file_id": file_id,
            "status": "not_found"
        }

    return {
        "file_id": file_id,
        "status": metadata.get("status", "unknown"),
        "original_filename": metadata.get("original_filename", "Unknown")
    }


@app.post("/ask")
def ask_question(req: QuestionRequest):
    """
    Ask a question about the uploaded PDF
    """
    # Check file status
    metadata = get_metadata_from_table(req.file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    current_status = metadata.get("status")

    if current_status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"File not ready. Current status: {current_status}"
        )

    print(f"❓ Question for {req.file_id}: {req.question}")

    # Create embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )

    # Initialize vectorstore
    from langchain_pinecone import PineconeVectorStore

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=req.file_id
    )

    # Get chunk count from metadata
    chunk_count = metadata.get("chunk_count", 0)
    # Use all chunks but cap at 50 for performance
    k_value = min(chunk_count, 50)

    print("K value:", k_value);

    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})

    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
            {
                "context": itemgetter("context") | RunnableLambda(format_docs),
                "question": itemgetter("question")
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    docs = retriever.invoke(req.question)
    print(f"✅ Retrieved {len(docs)} relevant chunks")

    answer = chain.invoke({
        "context": docs,
        "question": req.question
    })

    return {
        "answer": answer,
        "file_id": req.file_id,
        "original_filename": metadata.get("original_filename", "Unknown")
    }


@app.delete("/files/{file_id}")
def delete_file(file_id: str):
    """
    Delete file from Azure Blob, Pinecone, and Table Storage
    """
    try:
        # Delete from Pinecone
        try:
            index.delete(delete_all=True, namespace=file_id)
            print(f"🗑️ Deleted Pinecone namespace: {file_id}")
        except Exception as e:
            print(f"⚠️ Pinecone delete warning: {e}")

        # Delete from Azure Blob
        try:
            blob_client = container_client.get_blob_client(f"{file_id}.pdf")
            blob_client.delete_blob()
            print(f"🗑️ Deleted blob: {file_id}.pdf")
        except Exception as e:
            print(f"⚠️ Blob delete warning: {e}")

        # Delete from Table Storage
        try:
            table_client.delete_entity(partition_key="pdf_files", row_key=file_id)
            print(f"🗑️ Deleted metadata for {file_id}")
        except Exception as e:
            print(f"⚠️ Table delete warning: {e}")

        return {"message": "Deleted successfully", "file_id": file_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """
    Health check endpoint
    """
    file_count = len(list_all_files_from_table())
    return {
        "message": "PDF QA Backend v2.0 with Azure Table Storage",
        "total_files": file_count,
        "endpoints": {
            "upload": "POST /upload",
            "list_files": "GET /files",
            "get_file_metadata": "GET /files/{file_id}",
            "download_pdf": "GET /files/{file_id}/download",
            "ask_question": "POST /ask",
            "delete_file": "DELETE /files/{file_id}"
        }
    }

# =========================
# RUN
# =========================

# uvicorn main_enhanced:app --reload