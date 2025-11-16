"""Agent logs API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import uuid

from app.db.sync_database import get_agent_logs, delete_agent_logs
from app.schemas.agent_logs import AgentLogResponse, AgentLogDeleteResponse

router = APIRouter()


@router.get("/{pipeline_run_id}", response_model=List[AgentLogResponse])
async def get_pipeline_agent_logs(
    pipeline_run_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    level: Optional[str] = Query(default=None),
    agent_name: Optional[str] = Query(default=None)
):
    """Get agent logs for a specific pipeline run."""
    try:
        logs = get_agent_logs(
            pipeline_run_id=str(pipeline_run_id),
            limit=limit,
            offset=offset,
            level=level,
            agent_name=agent_name
        )
        
        # Convert to Pydantic models
        response_logs = []
        for log in logs:
            response_logs.append(AgentLogResponse(**log))
        
        return response_logs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch agent logs: {str(e)}")


@router.delete("/{pipeline_run_id}", response_model=AgentLogDeleteResponse)
async def delete_pipeline_agent_logs(pipeline_run_id: uuid.UUID):
    """Delete all agent logs for a specific pipeline run."""
    try:
        deleted_count = delete_agent_logs(str(pipeline_run_id))
        
        return AgentLogDeleteResponse(
            deleted_count=deleted_count,
            message=f"Successfully deleted {deleted_count} agent logs for pipeline {pipeline_run_id}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete agent logs: {str(e)}")