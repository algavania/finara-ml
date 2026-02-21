from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

from app.routers import xai, optimizer, profiler, parser

tags_metadata = [
    {
        "name": "XAI",
        "description": "Explainable Artificial Intelligence for Risk Assessment. Evaluates user debt risk and provides SHAP value explanations for transparency.",
    },
    {
        "name": "Parser",
        "description": "Smart document parsing using Gemini Multimodal to extract transactions from receipts and e-statements.",
    },
    {
        "name": "Optimizer",
        "description": "Reinforcement Learning (PPO) agent that recommends optimal debt payoff strategies. *(Phase 5)*",
    },
    {
        "name": "Profiler",
        "description": "Behavioral clustering to identify spending archetypes and generate personalized tips. *(Phase 5)*",
    },
]

app = FastAPI(
    title="Finara ML Engine",
    description="AI/ML microservice for explainable debt risk assessment, RL optimization, and behavioral profiling. \n\n**Note**: All endpoints require an API Key passed in the `X-API-Key` header.",
    version="0.1.0",
    openapi_tags=tags_metadata
)

# CORS — allow Edge Function and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("SUPABASE_URL", "http://localhost:54321"),
        "http://localhost:3000",
    ],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# API key auth
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    expected = os.getenv("ML_API_KEY")
    if not expected or api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Include routers
app.include_router(xai.router, prefix="/api/xai", tags=["XAI"], dependencies=[Depends(verify_api_key)])
app.include_router(parser.router, prefix="/api/parser", tags=["Parser"], dependencies=[Depends(verify_api_key)])
app.include_router(optimizer.router, prefix="/api/optimizer", tags=["Optimizer"], dependencies=[Depends(verify_api_key)])
app.include_router(profiler.router, prefix="/api/profiler", tags=["Profiler"], dependencies=[Depends(verify_api_key)])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "finara-ml"}
