from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import setup_qa_chain

app = FastAPI(title="DATAmat API")

class Query(BaseModel):
    question: str

class Response(BaseModel):
    answer: str

qa_chain = setup_qa_chain()

@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        result = qa_chain.invoke({"query": query.question})
        return Response(answer=result["result"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 