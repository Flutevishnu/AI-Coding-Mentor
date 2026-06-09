
import logging

from fastapi import FastAPI
from core.config import settings
from fastapi.middleware.cors import CORSMiddleware
from api import api_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG_MODE
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
@app.get("/")
async def root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME} API. Visit /docs for documentation."}