from fastapi import APIRouter, status
from core.config import settings
from schemas.response import HealthCheckResponse

health_router = APIRouter(tags=["System Health"])

@health_router.get(
    "/health", 
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Perform a system health check"
)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        version=settings.VERSION,
        debug=settings.DEBUG_MODE
    )