"""Pipeline service for managing ContentRunway pipeline runs."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import logging

from app.models.pipeline import PipelineRun, TopicIdea, ResearchSource, ContentDraft
from app.core.config import settings

logger = logging.getLogger(__name__)


class PipelineService:
    """Service for managing pipeline runs and their lifecycle."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def start_pipeline(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new content pipeline run."""
        try:
            # Create new pipeline run
            pipeline_run = PipelineRun(
                tenant_id=pipeline_data.get('tenant_id', 'personal'),
                domain_focus=pipeline_data['domain_focus'],
                quality_thresholds=pipeline_data.get('quality_thresholds', {
                    'overall': settings.QUALITY_THRESHOLD_OVERALL,
                    'technical': settings.QUALITY_THRESHOLD_TECHNICAL,
                    'domain_expertise': settings.QUALITY_THRESHOLD_DOMAIN,
                    'style_consistency': settings.QUALITY_THRESHOLD_STYLE,
                    'compliance': settings.QUALITY_THRESHOLD_COMPLIANCE
                }),
                status='initialized',
                current_step='initialized',
                progress_percentage=0.0,
                retry_count=0,
                max_retries=3
            )
            
            self.db.add(pipeline_run)
            await self.db.commit()
            await self.db.refresh(pipeline_run)
            
            # Store initial state in Redis for pipeline execution
            from app.services.redis_service import redis_service
            initial_state = {
                "run_id": str(pipeline_run.id),
                "tenant_id": pipeline_run.tenant_id,
                "status": pipeline_run.status,
                "domain_focus": pipeline_data['domain_focus'],
                "quality_thresholds": pipeline_run.quality_thresholds,
                "current_step": "initialized",
                "progress_percentage": 0.0,
                "created_at": pipeline_run.created_at
            }
            await redis_service.store_pipeline_state(str(pipeline_run.id), initial_state)
            
            # TODO: Trigger LangGraph pipeline execution
            # This would integrate with Celery to run the pipeline asynchronously
            # await self._trigger_pipeline_execution(pipeline_run.id, pipeline_data)
            
            return {
                'id': str(pipeline_run.id),
                'tenant_id': pipeline_run.tenant_id,
                'status': pipeline_run.status,
                'domain_focus': pipeline_run.domain_focus,
                'quality_thresholds': pipeline_run.quality_thresholds,
                'created_at': pipeline_run.created_at.isoformat(),
                'current_step': pipeline_run.current_step,
                'progress_percentage': pipeline_run.progress_percentage
            }
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to start pipeline: {e}")
            raise
    
    async def get_pipeline_run(self, run_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get a specific pipeline run by ID."""
        try:
            result = await self.db.execute(
                select(PipelineRun).where(PipelineRun.id == run_id)
            )
            pipeline_run = result.scalar_one_or_none()
            
            if not pipeline_run:
                return None
            
            return {
                'id': str(pipeline_run.id),
                'tenant_id': pipeline_run.tenant_id,
                'status': pipeline_run.status,
                'domain_focus': pipeline_run.domain_focus,
                'quality_thresholds': pipeline_run.quality_thresholds,
                'created_at': pipeline_run.created_at.isoformat(),
                'started_at': pipeline_run.started_at.isoformat() if pipeline_run.started_at else None,
                'completed_at': pipeline_run.completed_at.isoformat() if pipeline_run.completed_at else None,
                'current_step': pipeline_run.current_step,
                'progress_percentage': pipeline_run.progress_percentage,
                'chosen_topic_id': str(pipeline_run.chosen_topic_id) if pipeline_run.chosen_topic_id else None,
                'final_quality_score': pipeline_run.final_quality_score,
                'human_approved': pipeline_run.human_approved,
                'published_urls': pipeline_run.published_urls or [],
                'error_message': pipeline_run.error_message,
                'retry_count': pipeline_run.retry_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline run {run_id}: {e}")
            raise
    
    async def list_pipeline_runs(
        self,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None,
        tenant_id: str = 'personal'
    ) -> List[Dict[str, Any]]:
        """List pipeline runs with filtering options."""
        try:
            query = select(PipelineRun).where(PipelineRun.tenant_id == tenant_id)
            
            if status:
                query = query.where(PipelineRun.status == status)
            
            query = query.order_by(PipelineRun.created_at.desc()).limit(limit).offset(offset)
            
            result = await self.db.execute(query)
            pipeline_runs = result.scalars().all()
            
            return [
                {
                    'id': str(run.id),
                    'status': run.status,
                    'domain_focus': run.domain_focus,
                    'created_at': run.created_at.isoformat(),
                    'current_step': run.current_step,
                    'progress_percentage': run.progress_percentage,
                    'final_quality_score': run.final_quality_score,
                    'human_approved': run.human_approved,
                    'error_message': run.error_message
                }
                for run in pipeline_runs
            ]
            
        except Exception as e:
            logger.error(f"Failed to list pipeline runs: {e}")
            raise
    
    async def get_pipeline_status(self, run_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get current status of a pipeline run."""
        try:
            result = await self.db.execute(
                select(PipelineRun.status, PipelineRun.current_step, PipelineRun.progress_percentage,
                       PipelineRun.error_message).where(PipelineRun.id == run_id)
            )
            status_data = result.first()
            
            if not status_data:
                return None
            
            return {
                'status': status_data.status,
                'current_step': status_data.current_step,
                'progress_percentage': status_data.progress_percentage,
                'error_message': status_data.error_message
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status {run_id}: {e}")
            raise
    
    async def pause_pipeline(self, run_id: uuid.UUID) -> bool:
        """Pause a running pipeline."""
        try:
            result = await self.db.execute(
                select(PipelineRun).where(PipelineRun.id == run_id)
            )
            pipeline_run = result.scalar_one_or_none()
            
            if not pipeline_run or pipeline_run.status != 'running':
                return False
            
            pipeline_run.status = 'paused'
            await self.db.commit()
            
            # TODO: Signal LangGraph pipeline to pause
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause pipeline {run_id}: {e}")
            return False
    
    async def resume_pipeline(self, run_id: uuid.UUID) -> bool:
        """Resume a paused pipeline."""
        try:
            result = await self.db.execute(
                select(PipelineRun).where(PipelineRun.id == run_id)
            )
            pipeline_run = result.scalar_one_or_none()
            
            if not pipeline_run or pipeline_run.status != 'paused':
                return False
            
            pipeline_run.status = 'running'
            await self.db.commit()
            
            # TODO: Signal LangGraph pipeline to resume
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume pipeline {run_id}: {e}")
            return False
    
    async def cancel_pipeline(self, run_id: uuid.UUID) -> bool:
        """Cancel a pipeline run."""
        try:
            result = await self.db.execute(
                select(PipelineRun).where(PipelineRun.id == run_id)
            )
            pipeline_run = result.scalar_one_or_none()
            
            if not pipeline_run or pipeline_run.status in ['completed', 'failed']:
                return False
            
            pipeline_run.status = 'cancelled'
            pipeline_run.completed_at = datetime.now()
            await self.db.commit()
            
            # TODO: Signal LangGraph pipeline to cancel
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel pipeline {run_id}: {e}")
            return False
    
    async def get_pipeline_topics(self, run_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Get topic ideas for a pipeline run."""
        try:
            result = await self.db.execute(
                select(TopicIdea)
                .where(TopicIdea.pipeline_run_id == run_id)
                .order_by(TopicIdea.overall_score.desc())
            )
            topics = result.scalars().all()
            
            return [
                {
                    'id': str(topic.id),
                    'title': topic.title,
                    'description': topic.description,
                    'domain': topic.domain,
                    'relevance_score': topic.relevance_score,
                    'novelty_score': topic.novelty_score,
                    'seo_difficulty': topic.seo_difficulty,
                    'overall_score': topic.overall_score,
                    'target_keywords': topic.target_keywords,
                    'estimated_traffic': topic.estimated_traffic,
                    'competition_level': topic.competition_level,
                    'source_count': topic.source_count,
                    'is_selected': topic.is_selected
                }
                for topic in topics
            ]
            
        except Exception as e:
            logger.error(f"Failed to get pipeline topics {run_id}: {e}")
            raise
    
    async def select_topic(self, run_id: uuid.UUID, topic_id: uuid.UUID) -> bool:
        """Select a topic for content creation."""
        try:
            # First, unselect all topics for this pipeline
            await self.db.execute(
                select(TopicIdea)
                .where(TopicIdea.pipeline_run_id == run_id)
                .update({'is_selected': False})
            )
            
            # Select the chosen topic
            result = await self.db.execute(
                select(TopicIdea).where(
                    TopicIdea.pipeline_run_id == run_id,
                    TopicIdea.id == topic_id
                )
            )
            topic = result.scalar_one_or_none()
            
            if not topic:
                return False
            
            topic.is_selected = True
            
            # Update the pipeline run
            pipeline_result = await self.db.execute(
                select(PipelineRun).where(PipelineRun.id == run_id)
            )
            pipeline_run = pipeline_result.scalar_one_or_none()
            
            if pipeline_run:
                pipeline_run.chosen_topic_id = topic_id
            
            await self.db.commit()
            
            # TODO: Trigger next phase of pipeline
            
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to select topic {topic_id} for pipeline {run_id}: {e}")
            return False
    
    async def get_pipeline_stats(self, tenant_id: str = 'personal') -> Dict[str, Any]:
        """Get pipeline statistics for dashboard."""
        try:
            # Total runs
            total_result = await self.db.execute(
                select(func.count(PipelineRun.id))
                .where(PipelineRun.tenant_id == tenant_id)
            )
            total_runs = total_result.scalar() or 0
            
            # Status counts
            status_result = await self.db.execute(
                select(PipelineRun.status, func.count(PipelineRun.id))
                .where(PipelineRun.tenant_id == tenant_id)
                .group_by(PipelineRun.status)
            )
            status_counts = dict(status_result.fetchall())
            
            # Success rate
            completed_runs = status_counts.get('completed', 0)
            failed_runs = status_counts.get('failed', 0)
            success_rate = completed_runs / max(completed_runs + failed_runs, 1)
            
            # Average processing time (placeholder)
            avg_processing_time = 25.0  # minutes
            
            return {
                'total_runs': total_runs,
                'active_runs': status_counts.get('running', 0) + status_counts.get('paused', 0),
                'completed_runs': completed_runs,
                'failed_runs': failed_runs,
                'success_rate': success_rate,
                'avg_processing_time': avg_processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {e}")
            # Return default stats on error
            return {
                'total_runs': 0,
                'active_runs': 0,
                'completed_runs': 0,
                'failed_runs': 0,
                'success_rate': 0.0,
                'avg_processing_time': 0.0
            }