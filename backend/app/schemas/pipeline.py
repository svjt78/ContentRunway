"""Pydantic schemas for pipeline API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class PipelineRunCreate(BaseModel):
    """Schema for creating a new pipeline run."""
    research_query: str = Field(..., description="Research topic or query")
    domain_focus: List[str] = Field(..., description="List of domain focuses")
    quality_thresholds: Dict[str, float] = Field(
        default={
            "overall": 0.85,
            "technical": 0.90,
            "domain_expertise": 0.90,
            "style_consistency": 0.88,
            "compliance": 0.95
        },
        description="Quality threshold settings"
    )
    tenant_id: str = Field(default="personal", description="Tenant identifier")


class PipelineRunResponse(BaseModel):
    """Schema for pipeline run response."""
    id: str
    tenant_id: str
    status: str
    domain_focus: List[str]
    quality_thresholds: Dict[str, float]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: str
    progress_percentage: float
    chosen_topic_id: Optional[str] = None
    final_quality_score: Optional[float] = None
    human_approved: bool
    published_urls: List[str] = []
    error_message: Optional[str] = None
    retry_count: int = 0


class PipelineStatus(BaseModel):
    """Schema for pipeline status response."""
    status: str
    current_step: str
    progress_percentage: float
    error_message: Optional[str] = None


class TopicIdeaResponse(BaseModel):
    """Schema for topic idea response."""
    id: str
    title: str
    description: str
    domain: str
    relevance_score: float
    novelty_score: float
    seo_difficulty: float
    overall_score: float
    target_keywords: List[str]
    estimated_traffic: Optional[int] = None
    competition_level: Optional[str] = None
    source_count: int = 0
    is_selected: bool = False


class PipelineStatsResponse(BaseModel):
    """Schema for pipeline statistics."""
    total_runs: int
    active_runs: int
    completed_runs: int
    failed_runs: int
    success_rate: float
    avg_processing_time: float  # in minutes