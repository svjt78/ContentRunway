from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import uuid

from app.db.database import get_db
from app.services.pipeline_service import PipelineService
from app.schemas.pipeline import (
    PipelineRunCreate,
    PipelineRunResponse,
    PipelineStatus,
    TopicIdeaResponse
)

router = APIRouter()

@router.post("/start", response_model=PipelineRunResponse)
async def start_pipeline(
    pipeline_request: PipelineRunCreate,
    db: AsyncSession = Depends(get_db)
):
    """Start a new content pipeline run."""
    service = PipelineService(db)
    pipeline_run = await service.start_pipeline(pipeline_request)
    return pipeline_run

@router.get("/runs", response_model=List[PipelineRunResponse])
async def list_pipeline_runs(
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List pipeline runs with optional filtering."""
    service = PipelineService(db)
    runs = await service.list_pipeline_runs(limit=limit, offset=offset, status=status)
    return runs

@router.get("/runs/{run_id}", response_model=PipelineRunResponse)
async def get_pipeline_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific pipeline run by ID."""
    service = PipelineService(db)
    pipeline_run = await service.get_pipeline_run(run_id)
    if not pipeline_run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return pipeline_run

@router.get("/runs/{run_id}/status", response_model=PipelineStatus)
async def get_pipeline_status(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get current status of a pipeline run."""
    service = PipelineService(db)
    status = await service.get_pipeline_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return status

@router.post("/runs/{run_id}/pause")
async def pause_pipeline(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Pause a running pipeline."""
    service = PipelineService(db)
    success = await service.pause_pipeline(run_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not pause pipeline")
    return {"message": "Pipeline paused successfully"}

@router.post("/runs/{run_id}/resume")
async def resume_pipeline(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Resume a paused pipeline."""
    service = PipelineService(db)
    success = await service.resume_pipeline(run_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not resume pipeline")
    return {"message": "Pipeline resumed successfully"}

@router.delete("/runs/{run_id}")
async def cancel_pipeline(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Cancel a pipeline run."""
    service = PipelineService(db)
    success = await service.cancel_pipeline(run_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not cancel pipeline")
    return {"message": "Pipeline cancelled successfully"}

@router.get("/runs/{run_id}/topics", response_model=List[TopicIdeaResponse])
async def get_pipeline_topics(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get topic ideas generated for a pipeline run."""
    service = PipelineService(db)
    topics = await service.get_pipeline_topics(run_id)
    return topics

@router.post("/runs/{run_id}/topics/{topic_id}/select")
async def select_topic(
    run_id: uuid.UUID,
    topic_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Select a topic for content creation."""
    service = PipelineService(db)
    success = await service.select_topic(run_id, topic_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not select topic")
    return {"message": "Topic selected successfully"}

@router.get("/stats")
async def get_pipeline_stats(
    db: AsyncSession = Depends(get_db)
):
    """Get pipeline statistics for dashboard."""
    service = PipelineService(db)
    stats = await service.get_pipeline_stats()
    return stats

@router.get("/health")
async def pipeline_health_check(
    db: AsyncSession = Depends(get_db)
):
    """Health check endpoint to monitor database performance."""
    import time
    start_time = time.time()
    
    try:
        # Simple database query to test connection speed
        from sqlalchemy import text
        result = await db.execute(text("SELECT 1"))
        result.fetchone()
        
        db_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "database_response_time_ms": round(db_time * 1000, 2),
            "connection_pool_status": "ok"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "database_response_time_ms": round((time.time() - start_time) * 1000, 2)
        }