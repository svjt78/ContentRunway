"""
Background tasks for ContentRunway pipeline execution using Celery.
"""

import sys
import os
from typing import Dict, Any, Optional
import uuid
import asyncio
import logging
from datetime import datetime
from celery import current_task
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update

# Add the langgraph directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../langgraph'))

from app.worker import celery_app
from app.core.config import settings
from app.services.redis_service import redis_service
from app.models.pipeline import PipelineRun

# Import LangGraph pipeline - temporarily commented out for testing
# from contentrunway.pipeline import ContentPipeline
# from contentrunway.state.pipeline_state import ContentPipelineState, QualityScores

logger = logging.getLogger(__name__)

# Create async database engine for tasks - initialize later to avoid import errors
engine = None
AsyncSessionLocal = None

def get_async_engine():
    """Get or create async database engine."""
    global engine, AsyncSessionLocal
    if engine is None:
        # Convert postgres:// to postgresql+asyncpg:// for async support
        db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        engine = create_async_engine(db_url)
        AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine


@celery_app.task(bind=True, name="execute_pipeline")
def execute_content_pipeline(self, run_id: str, pipeline_config: Dict[str, Any]):
    """
    Execute the complete ContentRunway pipeline in the background.
    
    Args:
        run_id: UUID of the pipeline run
        pipeline_config: Configuration dictionary for the pipeline
    """
    try:
        logger.info(f"Starting pipeline execution for run_id: {run_id}")
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"run_id": run_id, "status": "starting", "progress": 5}
        )
        
        # Run the async pipeline execution
        result = asyncio.run(_execute_pipeline_async(run_id, pipeline_config, self))
        
        logger.info(f"Pipeline execution completed for run_id: {run_id}")
        return result
        
    except Exception as e:
        logger.error(f"Pipeline execution failed for run_id: {run_id} - {str(e)}")
        
        # Update database with failure
        asyncio.run(_update_pipeline_status(run_id, "failed", error_message=str(e)))
        
        # Update task state
        self.update_state(
            state="FAILURE",
            meta={"run_id": run_id, "error": str(e)}
        )
        
        raise e


async def _execute_pipeline_async(run_id: str, pipeline_config: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute the pipeline asynchronously - simplified for testing orchestration."""
    
    logger.info(f"Starting simplified pipeline execution for {run_id}")
    
    # Update database status to running
    await _update_pipeline_status(run_id, "running", current_step="research", progress_percentage=10.0)
    
    try:
        # Simulate pipeline execution stages
        stages = [
            ("research", 20.0),
            ("curation", 40.0), 
            ("writing", 60.0),
            ("quality_gates", 80.0),
            ("formatting", 90.0),
            ("publishing", 95.0),
            ("completed", 100.0)
        ]
        
        for stage, progress in stages:
            logger.info(f"Pipeline {run_id}: Executing stage {stage}")
            
            # Update progress
            await _update_pipeline_status(run_id, "running", current_step=stage, progress_percentage=progress)
            
            # Update Celery task state
            celery_task.update_state(
                state="PROGRESS",
                meta={
                    "run_id": run_id,
                    "status": "running",
                    "current_step": stage,
                    "progress": progress
                }
            )
            
            # Simulate work (shorter for testing)
            await asyncio.sleep(2)
        
        # Simulate final results
        final_state = {
            "run_id": run_id,
            "status": "completed",
            "published_urls": ["http://digitaldossier.us/documents/test-doc-1"],
            "processing_time": 10.0  # simulated
        }
        
        # Update database with completion
        await _update_pipeline_completion(run_id, final_state)
        
        # Update Celery task state
        celery_task.update_state(
            state="SUCCESS",
            meta={
                "run_id": run_id,
                "status": "completed",
                "progress": 100,
                "final_state": final_state
            }
        )
        
        logger.info(f"Pipeline {run_id} completed successfully")
        return final_state
        
    except Exception as e:
        logger.error(f"Pipeline execution error: {str(e)}")
        await _update_pipeline_status(run_id, "failed", error_message=str(e))
        raise


async def _update_pipeline_status(
    run_id: str, 
    status: str, 
    current_step: Optional[str] = None,
    progress_percentage: Optional[float] = None,
    error_message: Optional[str] = None
):
    """Update pipeline status in database."""
    get_async_engine()  # Initialize engine
    async with AsyncSessionLocal() as session:
        try:
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
            await session.execute(stmt)
            await session.commit()
            
            # Also update Redis state for real-time monitoring
            redis_state = {
                "run_id": run_id,
                "status": status,
                "current_step": current_step or "unknown",
                "progress_percentage": progress_percentage or 0.0,
                "error_message": error_message,
                "updated_at": datetime.now().isoformat()
            }
            await redis_service.store_pipeline_state(run_id, redis_state)
            
        except Exception as e:
            logger.error(f"Failed to update pipeline status: {e}")
            await session.rollback()


async def _update_pipeline_completion(run_id: str, final_state: Dict[str, Any]):
    """Update pipeline with completion data."""
    get_async_engine()  # Initialize engine
    async with AsyncSessionLocal() as session:
        try:
            stmt = update(PipelineRun).where(PipelineRun.id == uuid.UUID(run_id))
            
            update_data = {
                "status": "completed",
                "completed_at": datetime.now(),
                "current_step": "completed",
                "progress_percentage": 100.0,
                "published_urls": final_state.get("published_urls", []),
                "final_quality_score": None,  # Simplified for testing
                "human_approved": True  # Simplified for testing
            }
            
            stmt = stmt.values(**update_data)
            await session.execute(stmt)
            await session.commit()
            
            # Update Redis with final state
            await redis_service.store_pipeline_state(run_id, final_state)
            
        except Exception as e:
            logger.error(f"Failed to update pipeline completion: {e}")
            await session.rollback()


@celery_app.task(name="cleanup_pipeline_checkpoints")
def cleanup_pipeline_checkpoints(run_id: str):
    """Clean up LangGraph checkpoint files after pipeline completion."""
    try:
        checkpoint_file = f"pipeline_checkpoints_{run_id}.db"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info(f"Cleaned up checkpoint file: {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to cleanup checkpoint file: {e}")


# Task to monitor and recover stuck pipelines
@celery_app.task(name="monitor_pipeline_health")
def monitor_pipeline_health():
    """Monitor pipeline health and recover stuck processes."""
    try:
        # This would implement health checks for running pipelines
        # and recover any stuck processes
        logger.info("Pipeline health monitoring completed")
    except Exception as e:
        logger.error(f"Pipeline health monitoring failed: {e}")