from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, HttpUrl
from utils import setup_qa_chain, process_dataset
import os
from pathlib import Path
from datetime import datetime
import aiohttp
import aiofiles
from typing import Optional, List
import shutil

app = FastAPI(title="DATAmat API")

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

@app.post("/ask")
async def ask_question(query: Query):
    try:
        result = qa_chain.invoke({"query": query.question})
        return {"answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = DOWNLOAD_DIR / safe_filename
        
        # Save uploaded file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        finally:
            file.file.close()
        
        # Update the QA chain with the new dataset
        global qa_chain
        qa_chain = setup_qa_chain(force_reload=True)
        
        return {
            "message": "Dataset uploaded and processed successfully",
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": os.path.getsize(file_path)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading dataset: {str(e)}"
        )

@app.post("/download-dataset")
async def download_dataset(dataset: DatasetDownload):
    try:
        # Generate filename if not provided
        if not dataset.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(str(dataset.url)).suffix or '.csv'
            dataset.filename = f"dataset_{timestamp}{file_extension}"
        
        # Ensure filename is safe
        safe_filename = Path(dataset.filename).name
        file_path = DOWNLOAD_DIR / safe_filename
        
        # Download file
        async with aiohttp.ClientSession() as session:
            async with session.get(str(dataset.url)) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download file from URL. Status: {response.status}"
                    )
                
                content = await response.read()
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(content)
        
        # Update the QA chain with the new dataset
        global qa_chain
        qa_chain = setup_qa_chain(force_reload=True)
        
        return {
            "message": "Dataset downloaded and processed successfully",
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": os.path.getsize(file_path)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading dataset: {str(e)}"
        )

@app.get("/list-datasets")
async def list_datasets():
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
        raise HTTPException(
            status_code=500,
            detail=f"Error listing datasets: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 