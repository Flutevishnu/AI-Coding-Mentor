from fastapi import APIRouter
from .health import health_router
from .chat import chat_router
api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(chat_router)