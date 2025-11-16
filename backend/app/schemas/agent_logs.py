"""Pydantic schemas for agent logs API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class AgentLogResponse(BaseModel):
    """Schema for agent log response."""
    id: str
    pipeline_run_id: str
    agent_name: str
    stage: str
    operation: str
    message: str
    context: Optional[Dict[str, Any]] = None
    level: str = "INFO"
    timestamp: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class AgentLogListParams(BaseModel):
    """Schema for agent log list query parameters."""
    limit: int = Field(default=100, ge=1, le=1000, description="Number of logs to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    level: Optional[str] = Field(default=None, description="Filter by log level")
    agent_name: Optional[str] = Field(default=None, description="Filter by agent name")


class AgentLogDeleteResponse(BaseModel):
    """Schema for agent log delete response."""
    deleted_count: int
    message: str