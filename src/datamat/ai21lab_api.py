from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl
from src.datamat.ai21labs_utils import setup_qa_chain
import os
from pathlib import Path
from datetime import datetime
import aiohttp
import aiofiles
from typing import Optional, List
import shutil
import logging
from logger import logging
import kaggle

from fastapi.middleware.cors import CORSMiddleware
import shutil
import os





UPLOAD_DIR = "uploads"


# Configure logging for this module
logger = logging.getLogger(__name__)

app = FastAPI(title="DATAmat Ai21Lab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure download directory
DOWNLOAD_DIR = Path("datasets")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Initialize QA chain
qa_chain = setup_qa_chain()

class Query(BaseModel):
    question: str

class DatasetDownload(BaseModel):
    url: HttpUrl
    filename: Optional[str] = None

class KaggleDatasetDownload(BaseModel):
    dataset_name: str  # Format: "username/dataset-name"
    filename: Optional[str] = None

@app.post("/ai21/ask")
async def ask_question(query: Query):
    """Endpoint to ask questions using AI21 Labs LLM."""
    try:
        result = qa_chain.invoke({"query": query.question})
        return {"answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/ai21/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset and process it."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = DOWNLOAD_DIR / safe_filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        global qa_chain
        qa_chain = setup_qa_chain(force_reload=True)

        return {
            "message": "Dataset uploaded and processed successfully",
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": os.path.getsize(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")


@app.get("/ai21/list-datasets")
async def list_datasets():
    """List all available datasets."""
    try:
        files = []
        for file_path in DOWNLOAD_DIR.glob('*'):
            files.append({
                "filename": file_path.name,
                "size_bytes": os.path.getsize(file_path),
                "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
            })
        return {"datasets": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@app.get("/ai21/health")
async def health_check():
    """Health check endpoint."""
    try:
        if not os.getenv("AI21_API_KEY"):
            raise ValueError("AI21_API_KEY not found")
        return {"status": "healthy", "backend": "ai21"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/ai21/download-kaggle-dataset")
async def download_kaggle_dataset(dataset: KaggleDatasetDownload):
    """Download a dataset from Kaggle."""
    try:
        # Check for Kaggle credentials
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_cred_file = kaggle_dir / 'kaggle.json'
        
        if not kaggle_dir.exists():
            kaggle_dir.mkdir(parents=True)
            raise HTTPException(
                status_code=400,
                detail="Kaggle directory not found. Please create ~/.kaggle directory"
            )
            
        if not kaggle_cred_file.exists():
            raise HTTPException(
                status_code=400,
                detail="Kaggle API credentials not found. Please place kaggle.json in ~/.kaggle/"
            )

        # Validate dataset name format
        if "/" not in dataset.dataset_name:
            raise HTTPException(
                status_code=400,
                detail="Invalid dataset name format. Use 'username/dataset-name'"
            )

        download_path = DOWNLOAD_DIR / "kaggle" / dataset.dataset_name.replace("/", "_")
        download_path.mkdir(parents=True, exist_ok=True)

        try:
            kaggle.api.dataset_download_files(
                dataset.dataset_name,
                path=str(download_path),
                unzip=True
            )
        except Exception as kaggle_error:
            raise HTTPException(
                status_code=500,
                detail=f"Kaggle API error: {str(kaggle_error)}"
            )

        downloaded_files = list(download_path.glob('*'))
        if not downloaded_files:
            raise HTTPException(
                status_code=500,
                detail="No files were downloaded from Kaggle"
            )

        # Reload QA chain only if files were successfully downloaded
        global qa_chain
        qa_chain = setup_qa_chain(force_reload=True)

        return {
            "message": "Kaggle dataset downloaded successfully",
            "dataset": dataset.dataset_name,
            "files": [str(f.name) for f in downloaded_files],
            "download_path": str(download_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading Kaggle dataset: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)