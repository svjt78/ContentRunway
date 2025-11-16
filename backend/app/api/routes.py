from fastapi import APIRouter

from app.api.endpoints import pipeline, content, quality, review, publishing, agent_logs, quality_analytics

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    pipeline.router, 
    prefix="/pipeline", 
    tags=["pipeline"]
)

api_router.include_router(
    content.router, 
    prefix="/content", 
    tags=["content"]
)

api_router.include_router(
    quality.router, 
    prefix="/quality", 
    tags=["quality"]
)

api_router.include_router(
    review.router, 
    prefix="/review", 
    tags=["review"]
)

api_router.include_router(
    publishing.router, 
    prefix="/publishing", 
    tags=["publishing"]
)

api_router.include_router(
    agent_logs.router,
    prefix="/agent-logs",
    tags=["agent-logs"]
)

api_router.include_router(
    quality_analytics.router,
    prefix="/quality-analytics",
    tags=["quality-analytics"]
)