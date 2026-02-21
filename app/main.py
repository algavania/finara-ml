from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import os

from app.routers import xai, optimizer, profiler

app = FastAPI(
    title="Finara ML Engine",
    description="AI/ML microservice for explainable debt risk assessment, RL optimization, and behavioral profiling",
    version="0.1.0",
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
app.include_router(optimizer.router, prefix="/api/optimizer", tags=["Optimizer"], dependencies=[Depends(verify_api_key)])
app.include_router(profiler.router, prefix="/api/profiler", tags=["Profiler"], dependencies=[Depends(verify_api_key)])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "finara-ml"}
