"""API endpoints for pipeline status monitoring using Redis."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from app.services.redis_service import redis_service
from app.core.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline_status"])


@router.get("/status/{run_id}")
async def get_pipeline_status(
    run_id: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current pipeline status from Redis cache."""
    try:
        # Get lightweight checkpoint first
        checkpoint = await redis_service.get_pipeline_checkpoint(run_id)
        
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Pipeline run not found")
        
        # Verify user has access to this pipeline
        if checkpoint.get("tenant_id") != current_user.get("tenant_id"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "run_id": run_id,
            "status": checkpoint.get("status"),
            "current_step": checkpoint.get("current_step"),
            "progress_percentage": checkpoint.get("progress_percentage"),
            "updated_at": checkpoint.get("updated_at"),
            "error_message": checkpoint.get("error_message")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline status for {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve pipeline status")


@router.get("/full-state/{run_id}")
async def get_full_pipeline_state(
    run_id: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get complete pipeline state from Redis."""
    try:
        # Get full state
        full_state = await redis_service.get_pipeline_state(run_id)
        
        if not full_state:
            raise HTTPException(status_code=404, detail="Pipeline state not found")
        
        # Verify user has access
        if full_state.get("tenant_id") != current_user.get("tenant_id"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return full_state
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get full pipeline state for {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve pipeline state")


@router.get("/stats")
async def get_redis_stats(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get Redis usage statistics."""
    try:
        stats = await redis_service.get_stats()
        return {
            "redis_stats": stats,
            "connection_status": "connected" if redis_service.client else "disconnected"
        }
        
    except Exception as e:
        logger.error(f"Failed to get Redis stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve Redis statistics")


@router.post("/pause/{run_id}")
async def pause_pipeline(
    run_id: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Pause a running pipeline."""
    try:
        # Get current state
        current_state = await redis_service.get_pipeline_state(run_id)
        
        if not current_state:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Verify user has access
        if current_state.get("tenant_id") != current_user.get("tenant_id"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update status to paused
        current_state["status"] = "paused"
        current_state["paused_at"] = datetime.now().isoformat()
        
        # Store updated state
        success = await redis_service.store_pipeline_state(run_id, current_state)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to pause pipeline")
        
        return {
            "run_id": run_id,
            "status": "paused",
            "message": "Pipeline paused successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause pipeline {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to pause pipeline")


@router.post("/resume/{run_id}")
async def resume_pipeline(
    run_id: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Resume a paused pipeline."""
    try:
        # Get current state
        current_state = await redis_service.get_pipeline_state(run_id)
        
        if not current_state:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Verify user has access
        if current_state.get("tenant_id") != current_user.get("tenant_id"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if pipeline can be resumed
        if current_state.get("status") != "paused":
            raise HTTPException(
                status_code=400, 
                detail=f"Pipeline is {current_state.get('status')}, cannot resume"
            )
        
        # Update status to running
        current_state["status"] = "running"
        current_state["resumed_at"] = datetime.now().isoformat()
        
        # Store updated state
        success = await redis_service.store_pipeline_state(run_id, current_state)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to resume pipeline")
        
        # TODO: Trigger pipeline resumption in LangGraph
        # This would restart the pipeline from the current step
        
        return {
            "run_id": run_id,
            "status": "running",
            "current_step": current_state.get("current_step"),
            "message": "Pipeline resumed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume pipeline {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume pipeline")


@router.delete("/cache/{run_id}")
async def clear_pipeline_cache(
    run_id: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Clear cached data for a specific pipeline run."""
    try:
        # Verify pipeline exists and user has access
        current_state = await redis_service.get_pipeline_state(run_id)
        
        if current_state and current_state.get("tenant_id") != current_user.get("tenant_id"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete pipeline state
        state_deleted = await redis_service.delete("pipeline", f"full_state:{run_id}")
        checkpoint_deleted = await redis_service.delete("pipeline", f"checkpoint:{run_id}")
        
        return {
            "run_id": run_id,
            "state_deleted": state_deleted,
            "checkpoint_deleted": checkpoint_deleted,
            "message": "Pipeline cache cleared"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache for pipeline {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear pipeline cache")