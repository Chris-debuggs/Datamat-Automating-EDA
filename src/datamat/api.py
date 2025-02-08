from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from utils import setup_qa_chain
import os
from pathlib import Path
from datetime import datetime
import aiohttp
import aiofiles
from typing import Optional
import ssl
import certifi

app = FastAPI(title="DATAmat API")

# Model for dataset download request
class DatasetDownload(BaseModel):
    url: HttpUrl
    filename: Optional[str] = None

# Configure download directory
DOWNLOAD_DIR = Path("datasets")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Initialize QA chain
qa_chain = setup_qa_chain()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        result = qa_chain.invoke({"query": query.question})
        return {"answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Configure SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Download file asynchronously with SSL context
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(str(dataset.url), ssl=ssl_context) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download file from URL. Status: {response.status}"
                    )
                
                # Save file
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(await response.read())

        # Update the QA chain with the new dataset
        global qa_chain
        qa_chain = setup_qa_chain(force_reload=True)  # Force reload with new dataset
        
        return {
            "message": "Dataset downloaded and model updated successfully",
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