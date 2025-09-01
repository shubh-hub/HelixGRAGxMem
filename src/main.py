from fastapi import FastAPI
from src.api.v1.endpoints import search
from src.config import settings

# Initialize the FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Include the API router for search endpoints
app.include_router(search.router, prefix=settings.API_V1_STR, tags=["Search"])

@app.get("/", tags=["Health Check"])
def read_root():
    """
    Root endpoint for health checks.
    """
    return {"status": "ok", "message": f"Welcome to {settings.PROJECT_NAME}"}
