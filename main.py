from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from parser import parse
from estimator import estimate

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    repo_url: str

class AnalyzeResponse(BaseModel):
    metrics: dict
    issues: list[dict]
    carbon: dict

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    report = parse(req.repo_url)
    carbon = estimate(report['metrics']['total_bytes'])
    return {"metrics": report['metrics'], "issues": report['issues'], "carbon": carbon}
