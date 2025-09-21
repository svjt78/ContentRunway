"""
Synchronous database operations for Celery workers.

This module provides sync database operations that work correctly in
Celery's multiprocess fork environment, avoiding the "another operation
is in progress" errors that occur with async sessions.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from sqlalchemy import create_engine, text, select, update, insert
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.models.pipeline import PipelineRun
from app.models.pipeline import TopicIdea, ContentDraft, ResearchSource
from app.models.quality import QualityAssessment

logger = logging.getLogger(__name__)

# Sync database engine for worker processes
_sync_engine = None
_SyncSessionLocal = None

def get_sync_engine():
    """Get or create synchronous database engine for worker processes."""
    global _sync_engine, _SyncSessionLocal
    if _sync_engine is None:
        # Use standard postgresql:// URL for sync operations
        db_url = settings.DATABASE_URL
        if "postgresql+asyncpg://" in db_url:
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        
        _sync_engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=300
        )
        _SyncSessionLocal = sessionmaker(bind=_sync_engine)
        logger.info("Sync database engine initialized for worker processes")
    
    return _sync_engine

@contextmanager
def get_sync_session():
    """Get sync database session with proper cleanup."""
    get_sync_engine()  # Initialize if needed
    session = _SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

def update_pipeline_status(
    run_id: str, 
    status: str, 
    current_step: Optional[str] = None,
    progress_percentage: Optional[float] = None,
    error_message: Optional[str] = None
) -> bool:
    """Update pipeline status in database using sync operations."""
    try:
        with get_sync_session() as session:
            stmt = update(PipelineRun).where(PipelineRun.id == uuid.UUID(run_id))
            
            update_data = {"status": status}
            
            if current_step:
                update_data["current_step"] = current_step
            if progress_percentage is not None:
                update_data["progress_percentage"] = progress_percentage
            if error_message:
                update_data["error_message"] = error_message
            if status == "running" and not current_step:
                update_data["started_at"] = datetime.now()
            elif status in ["completed", "failed", "cancelled"]:
                update_data["completed_at"] = datetime.now()
                
            stmt = stmt.values(**update_data)
            result = session.execute(stmt)
            
            logger.info(f"Updated pipeline {run_id} status to {status}")
            return result.rowcount > 0
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to update pipeline status: {e}")
        return False

def update_pipeline_completion(run_id: str, final_state: Dict[str, Any]) -> bool:
    """Update pipeline with completion data using sync operations."""
    try:
        with get_sync_session() as session:
            stmt = update(PipelineRun).where(PipelineRun.id == uuid.UUID(run_id))
            
            update_data = {
                "status": "completed",
                "completed_at": datetime.now(),
                "current_step": "completed",
                "progress_percentage": 100.0,
                "published_urls": final_state.get("published_urls", []),
                "final_quality_score": final_state.get("final_quality_score"),
                "human_approved": final_state.get("human_approved", False)
            }
            
            stmt = stmt.values(**update_data)
            result = session.execute(stmt)
            
            logger.info(f"Updated pipeline {run_id} completion")
            return result.rowcount > 0
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to update pipeline completion: {e}")
        return False

def create_topic_idea(
    pipeline_run_id: str,
    title: str,
    description: str,
    domain: str,
    relevance_score: float,
    novelty_score: float,
    seo_difficulty: float,
    overall_score: float,
    keywords: List[str],
    is_selected: bool = True
) -> Optional[str]:
    """Create a topic idea and return its ID."""
    try:
        with get_sync_session() as session:
            topic_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "title": title,
                "description": description,
                "domain": domain,
                "relevance_score": relevance_score,
                "novelty_score": novelty_score,
                "seo_difficulty": seo_difficulty,
                "overall_score": overall_score,
                "target_keywords": keywords,
                "is_selected": is_selected
            }
            
            stmt = insert(TopicIdea).values(**topic_data).returning(TopicIdea.id)
            result = session.execute(stmt)
            topic_id = result.scalar()
            
            logger.info(f"Created topic idea {topic_id} for pipeline {pipeline_run_id}")
            return str(topic_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create topic idea: {e}")
        return None

def get_selected_topic_id(pipeline_run_id: str) -> Optional[str]:
    """Get the selected topic ID for a pipeline run."""
    try:
        with get_sync_session() as session:
            stmt = select(TopicIdea.id).where(
                TopicIdea.pipeline_run_id == uuid.UUID(pipeline_run_id),
                TopicIdea.is_selected == True
            ).limit(1)
            
            result = session.execute(stmt)
            topic_id = result.scalar()
            
            return str(topic_id) if topic_id else None
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to get selected topic: {e}")
        return None

def create_content_draft(
    pipeline_run_id: str,
    topic_id: str,
    title: str,
    content: str,
    **kwargs
) -> Optional[str]:
    """Create a content draft and return its ID."""
    try:
        with get_sync_session() as session:
            content_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "topic_id": uuid.UUID(topic_id),
                "version": kwargs.get("version", 1),
                "stage": kwargs.get("stage", "generated"),
                "title": title,
                "subtitle": kwargs.get("subtitle"),
                "abstract": kwargs.get("abstract"),
                "content": content,
                "citations": kwargs.get("citations", []),
                "word_count": kwargs.get("word_count", len(content.split())),
                "reading_time_minutes": kwargs.get("reading_time", max(1, len(content.split()) // 200)),
                "readability_score": kwargs.get("readability_score", 75.0),
                "meta_description": kwargs.get("meta_description"),
                "keywords": kwargs.get("keywords", []),
                "tags": kwargs.get("tags", [])
            }
            
            stmt = insert(ContentDraft).values(**content_data).returning(ContentDraft.id)
            result = session.execute(stmt)
            content_id = result.scalar()
            
            logger.info(f"Created content draft {content_id} for pipeline {pipeline_run_id}")
            return str(content_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create content draft: {e}")
        return None

def create_quality_assessment(
    pipeline_run_id: str,
    content_draft_id: str,
    gate_type: str,
    score: float,
    feedback: str,
    passed: bool,
    **kwargs
) -> Optional[str]:
    """Create a quality assessment and return its ID."""
    try:
        with get_sync_session() as session:
            assessment_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "content_draft_id": uuid.UUID(content_draft_id),
                "assessment_type": gate_type,
                "overall_score": score,
                "feedback": feedback,
                "passed": passed,
                "threshold": kwargs.get("threshold", 0.85),
                "details": kwargs.get("details", {}),
                "agent_version": kwargs.get("agent_version", "1.0"),
                "processing_time_seconds": kwargs.get("execution_time", 0.0)
            }
            
            stmt = insert(QualityAssessment).values(**assessment_data).returning(QualityAssessment.id)
            result = session.execute(stmt)
            result_id = result.scalar()
            
            logger.info(f"Created quality assessment {result_id} for {gate_type}")
            return str(result_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create quality assessment: {e}")
        return None

def create_research_source(
    pipeline_run_id: str,
    url: str,
    title: str,
    content: str,
    **kwargs
) -> Optional[str]:
    """Create a research source and return its ID."""
    try:
        with get_sync_session() as session:
            source_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "url": url,
                "title": title,
                "content": content,
                "domain": kwargs.get("domain"),
                "source_type": kwargs.get("source_type", "web"),
                "credibility_score": kwargs.get("credibility_score", 0.8),
                "relevance_score": kwargs.get("relevance_score", 0.8),
                "extracted_facts": kwargs.get("extracted_facts", []),
                "key_quotes": kwargs.get("key_quotes", []),
                "summary": kwargs.get("summary"),
                "word_count": kwargs.get("word_count", len(content.split())),
                "language": kwargs.get("language", "en"),
                "is_primary_source": kwargs.get("is_primary_source", False)
            }
            
            stmt = insert(ResearchSource).values(**source_data).returning(ResearchSource.id)
            result = session.execute(stmt)
            source_id = result.scalar()
            
            logger.info(f"Created research source {source_id} for pipeline {pipeline_run_id}")
            return str(source_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create research source: {e}")
        return None

def get_pipeline_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Get pipeline run details."""
    try:
        with get_sync_session() as session:
            stmt = select(PipelineRun).where(PipelineRun.id == uuid.UUID(run_id))
            result = session.execute(stmt)
            pipeline = result.scalar_one_or_none()
            
            if pipeline:
                return {
                    "id": str(pipeline.id),
                    "status": pipeline.status,
                    "current_step": pipeline.current_step,
                    "progress_percentage": pipeline.progress_percentage,
                    "error_message": pipeline.error_message,
                    "started_at": pipeline.started_at,
                    "completed_at": pipeline.completed_at,
                    "published_urls": pipeline.published_urls,
                    "final_quality_score": pipeline.final_quality_score,
                    "human_approved": pipeline.human_approved
                }
            return None
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to get pipeline run: {e}")
        return None

def test_sync_connection() -> bool:
    """Test the sync database connection."""
    try:
        with get_sync_session() as session:
            result = session.execute(text("SELECT 1 as test"))
            test_value = result.scalar()
            logger.info(f"Sync database connection test: {test_value}")
            return test_value == 1
    except Exception as e:
        logger.error(f"Sync database connection test failed: {e}")
        return False