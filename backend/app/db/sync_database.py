"""
Synchronous database operations for Celery workers.

This module provides sync database operations that work correctly in
Celery's multiprocess fork environment, avoiding the "another operation
is in progress" errors that occur with async sessions.
"""

import uuid
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from sqlalchemy import create_engine, text, select, update, insert
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.models.pipeline import PipelineRun
from app.models.pipeline import TopicIdea, ContentDraft, ResearchSource
from app.models.quality import QualityAssessment, FactCheckReport, DomainExpertiseReport, StyleAssessment, ComplianceReport, CritiqueReport
from app.models.content import ChannelContent, ContentOutline
from app.models.review import HumanReview, ReviewEdit
from app.models.publishing import Publication, PublishingAccount

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
    error_message: Optional[str] = None,
    review_session_id: Optional[str] = None
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
            if review_session_id:
                update_data["review_session_id"] = uuid.UUID(review_session_id)
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
    gate_name: str,
    overall_score: float,
    passed: bool,
    threshold_used: float,
    **kwargs
) -> Optional[str]:
    """Create a quality assessment and return its ID."""
    try:
        with get_sync_session() as session:
            assessment_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "content_draft_id": uuid.UUID(content_draft_id),
                "gate_name": gate_name,
                "assessor_type": kwargs.get("assessor_type", "ai_agent"),
                "assessment_version": kwargs.get("assessment_version", 1),
                "overall_score": overall_score,
                "passed": passed,
                "threshold_used": threshold_used,
                "criteria_scores": kwargs.get("criteria_scores", {}),
                "strengths": kwargs.get("strengths", []),
                "weaknesses": kwargs.get("weaknesses", []),
                "suggestions": kwargs.get("suggestions", []),
                "evidence": kwargs.get("evidence"),
                "reasoning": kwargs.get("reasoning"),
                "processing_time_seconds": kwargs.get("processing_time_seconds"),
                "model_used": kwargs.get("model_used"),
                "cost_estimate": kwargs.get("cost_estimate"),
                "requires_revision": kwargs.get("requires_revision", not passed),
                "revision_priority": kwargs.get("revision_priority", "medium")
            }
            
            stmt = insert(QualityAssessment).values(**assessment_data).returning(QualityAssessment.id)
            result = session.execute(stmt)
            result_id = result.scalar()
            
            logger.info(f"Created quality assessment {result_id} for {gate_name}")
            return str(result_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create quality assessment: {e}")
        return None

def create_research_source(
    pipeline_run_id: str,
    url: str,
    title: str,
    summary: str,
    **kwargs
) -> Optional[str]:
    """Create a research source and return its ID."""
    try:
        with get_sync_session() as session:
            source_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "topic_id": uuid.UUID(kwargs.get("topic_id")) if kwargs.get("topic_id") else None,
                "url": url,
                "title": title,
                "author": kwargs.get("author"),
                "publication_date": kwargs.get("publication_date"),
                "domain": kwargs.get("domain", "unknown"),
                "source_type": kwargs.get("source_type", "web"),
                "summary": summary,
                "key_points": kwargs.get("key_points", []),
                "quotable_content": kwargs.get("quotable_content", {}),
                "credibility_score": kwargs.get("credibility_score", 0.8),
                "relevance_score": kwargs.get("relevance_score", 0.8),
                "currency_score": kwargs.get("currency_score", 0.8),
                "citation_count": kwargs.get("citation_count", 0),
                "used_in_content": kwargs.get("used_in_content", False)
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

# Additional helper functions for comprehensive table coverage
# All functions include pipeline_run_id for proper pipeline tracking

def create_content_outline(
    pipeline_run_id: str,
    topic_id: str,
    sections: List[Dict],
    estimated_word_count: int,
    target_audience: str,
    primary_angle: str,
    key_takeaways: List[str],
    primary_keyword: str,
    secondary_keywords: List[str],
    **kwargs
) -> Optional[str]:
    """Create a content outline and return its ID."""
    try:
        with get_sync_session() as session:
            outline_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "topic_id": uuid.UUID(topic_id),
                "sections": sections,
                "estimated_word_count": estimated_word_count,
                "target_audience": target_audience,
                "primary_angle": primary_angle,
                "key_takeaways": key_takeaways,
                "call_to_action": kwargs.get("call_to_action"),
                "primary_keyword": primary_keyword,
                "secondary_keywords": secondary_keywords,
                "internal_link_opportunities": kwargs.get("internal_link_opportunities", []),
                "approved": kwargs.get("approved", False)
            }
            
            stmt = insert(ContentOutline).values(**outline_data).returning(ContentOutline.id)
            result = session.execute(stmt)
            outline_id = result.scalar()
            
            logger.info(f"Created content outline {outline_id} for pipeline {pipeline_run_id}")
            return str(outline_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create content outline: {e}")
        return None

def create_channel_content(
    pipeline_run_id: str,
    content_draft_id: str,
    platform: str,
    title: str,
    content: str,
    **kwargs
) -> Optional[str]:
    """Create channel-specific content and return its ID."""
    try:
        with get_sync_session() as session:
            channel_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "content_draft_id": uuid.UUID(content_draft_id),
                "platform": platform,
                "platform_specific_id": kwargs.get("platform_specific_id"),
                "title": title,
                "content": content,
                "excerpt": kwargs.get("excerpt"),
                "tags": kwargs.get("tags", []),
                "categories": kwargs.get("categories", []),
                "custom_fields": kwargs.get("custom_fields", {}),
                "include_toc": kwargs.get("include_toc", True),
                "include_citations": kwargs.get("include_citations", True),
                "formatting_style": kwargs.get("formatting_style", "standard"),
                "scheduled_publish_at": kwargs.get("scheduled_publish_at"),
                "canonical_url": kwargs.get("canonical_url"),
                "is_published": kwargs.get("is_published", False)
            }
            
            stmt = insert(ChannelContent).values(**channel_data).returning(ChannelContent.id)
            result = session.execute(stmt)
            channel_id = result.scalar()
            
            logger.info(f"Created channel content {channel_id} for pipeline {pipeline_run_id}")
            return str(channel_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create channel content: {e}")
        return None

def create_human_review(
    pipeline_run_id: str,
    content_draft_id: str,
    reviewer_id: str = "personal",
    **kwargs
) -> Optional[str]:
    """Create a human review session and return its ID."""
    try:
        with get_sync_session() as session:
            review_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "content_draft_id": uuid.UUID(content_draft_id),
                "reviewer_id": reviewer_id,
                "review_session_url": kwargs.get("review_session_url"),
                "status": kwargs.get("status", "pending"),
                "decision": kwargs.get("decision"),
                "started_at": kwargs.get("started_at"),
                "completed_at": kwargs.get("completed_at"),
                "time_spent_seconds": kwargs.get("time_spent_seconds"),
                "target_time_seconds": kwargs.get("target_time_seconds", 900),
                "overall_rating": kwargs.get("overall_rating"),
                "feedback_notes": kwargs.get("feedback_notes"),
                "checklist_items": kwargs.get("checklist_items", {}),
                "quality_concerns": kwargs.get("quality_concerns", []),
                "inline_edits": kwargs.get("inline_edits", []),
                "structural_changes": kwargs.get("structural_changes", []),
                "ai_recommendations_shown": kwargs.get("ai_recommendations_shown", []),
                "ai_recommendations_accepted": kwargs.get("ai_recommendations_accepted", [])
            }
            
            stmt = insert(HumanReview).values(**review_data).returning(HumanReview.id)
            result = session.execute(stmt)
            review_id = result.scalar()
            
            logger.info(f"Created human review {review_id} for pipeline {pipeline_run_id}")
            return str(review_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create human review: {e}")
        return None

def create_publication(
    pipeline_run_id: str,
    channel_content_id: str,
    platform: str,
    published_url: str,
    title: str,
    **kwargs
) -> Optional[str]:
    """Create a publication record and return its ID."""
    try:
        with get_sync_session() as session:
            publication_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "channel_content_id": uuid.UUID(channel_content_id),
                "platform": platform,
                "platform_content_id": kwargs.get("platform_content_id"),
                "published_url": published_url,
                "canonical_url": kwargs.get("canonical_url"),
                "title": title,
                "slug": kwargs.get("slug"),
                "status": kwargs.get("status", "published"),
                "visibility": kwargs.get("visibility", "public"),
                "scheduled_at": kwargs.get("scheduled_at"),
                "published_at": kwargs.get("published_at"),
                "last_updated_at": kwargs.get("last_updated_at"),
                "platform_metadata": kwargs.get("platform_metadata", {}),
                "tags": kwargs.get("tags", []),
                "categories": kwargs.get("categories", []),
                "publication_response": kwargs.get("publication_response", {}),
                "error_message": kwargs.get("error_message"),
                "retry_count": kwargs.get("retry_count", 0),
                "analytics_enabled": kwargs.get("analytics_enabled", True),
                "tracking_codes": kwargs.get("tracking_codes", {})
            }
            
            stmt = insert(Publication).values(**publication_data).returning(Publication.id)
            result = session.execute(stmt)
            publication_id = result.scalar()
            
            logger.info(f"Created publication {publication_id} for pipeline {pipeline_run_id}")
            return str(publication_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create publication: {e}")
        return None

def create_fact_check_report(
    pipeline_run_id: str,
    quality_assessment_id: str,
    content_draft_id: str,
    total_claims: int,
    verified_claims: int,
    disputed_claims: int,
    unverifiable_claims: int,
    claims_analysis: List[Dict],
    **kwargs
) -> Optional[str]:
    """Create a fact check report and return its ID."""
    try:
        with get_sync_session() as session:
            # Note: FactCheckReport doesn't have pipeline_run_id in schema, but we track it via quality_assessment
            report_data = {
                "quality_assessment_id": uuid.UUID(quality_assessment_id),
                "content_draft_id": uuid.UUID(content_draft_id),
                "total_claims": total_claims,
                "verified_claims": verified_claims,
                "disputed_claims": disputed_claims,
                "unverifiable_claims": unverifiable_claims,
                "claims_analysis": claims_analysis,
                "sources_checked": kwargs.get("sources_checked", 0),
                "reliable_sources": kwargs.get("reliable_sources", 0),
                "questionable_sources": kwargs.get("questionable_sources", 0),
                "supporting_evidence": kwargs.get("supporting_evidence", []),
                "contradicting_evidence": kwargs.get("contradicting_evidence", []),
                "corrections_needed": kwargs.get("corrections_needed", []),
                "additional_sources_suggested": kwargs.get("additional_sources_suggested", [])
            }
            
            stmt = insert(FactCheckReport).values(**report_data).returning(FactCheckReport.id)
            result = session.execute(stmt)
            report_id = result.scalar()
            
            logger.info(f"Created fact check report {report_id} for pipeline {pipeline_run_id}")
            return str(report_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create fact check report: {e}")
        return None

def create_critique_report(
    pipeline_run_id: str,
    content_draft_id: str,
    critique_cycle: int,
    overall_critique_score: float,
    pre_edit_quality_scores: Dict,
    post_edit_quality_scores: Dict,
    improvement_effectiveness: float,
    critique_feedback: Dict,
    **kwargs
) -> Optional[str]:
    """Create a critique report and return its ID."""
    try:
        with get_sync_session() as session:
            critique_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "content_draft_id": uuid.UUID(content_draft_id),
                "critique_cycle": critique_cycle,
                "cycle_type": kwargs.get("cycle_type", "initial"),
                "overall_critique_score": overall_critique_score,
                "pre_edit_quality_scores": pre_edit_quality_scores,
                "post_edit_quality_scores": post_edit_quality_scores,
                "improvement_effectiveness": improvement_effectiveness,
                "critique_feedback": critique_feedback,
                "issues_identified": kwargs.get("issues_identified", []),
                "issues_resolved": kwargs.get("issues_resolved", []),
                "issues_remaining": kwargs.get("issues_remaining", []),
                "editing_effectiveness": kwargs.get("editing_effectiveness", {}),
                "quality_gate_accuracy": kwargs.get("quality_gate_accuracy", {}),
                "retry_decision": kwargs.get("retry_decision", "pass"),
                "retry_reasoning": kwargs.get("retry_reasoning"),
                "next_action_required": kwargs.get("next_action_required"),
                "processing_time_seconds": kwargs.get("processing_time_seconds"),
                "model_used": kwargs.get("model_used"),
                "cost_estimate": kwargs.get("cost_estimate")
            }
            
            stmt = insert(CritiqueReport).values(**critique_data).returning(CritiqueReport.id)
            result = session.execute(stmt)
            critique_id = result.scalar()
            
            logger.info(f"Created critique report {critique_id} for pipeline {pipeline_run_id}")
            return str(critique_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create critique report: {e}")
        return None

def create_agent_log(
    pipeline_run_id: str,
    agent_name: str,
    stage: str,
    operation: str,
    message: str,
    level: str = "INFO",
    context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Create a new agent log entry and return its ID."""
    try:
        with get_sync_session() as session:
            log_data = {
                "pipeline_run_id": uuid.UUID(pipeline_run_id),
                "agent_name": agent_name,
                "stage": stage,
                "operation": operation,
                "message": message,
                "level": level,
                "context": context or {},
                "timestamp": datetime.now(),
                "created_at": datetime.now()
            }
            
            # Use raw SQL since we don't have a SQLAlchemy model yet
            stmt = text("""
                INSERT INTO agent_logs (
                    pipeline_run_id, agent_name, stage, operation, message, 
                    level, context, timestamp, created_at
                ) VALUES (
                    :pipeline_run_id, :agent_name, :stage, :operation, :message,
                    :level, :context, :timestamp, :created_at
                ) RETURNING id
            """)
            
            result = session.execute(stmt, {
                "pipeline_run_id": str(log_data["pipeline_run_id"]),
                "agent_name": log_data["agent_name"],
                "stage": log_data["stage"],
                "operation": log_data["operation"],
                "message": log_data["message"],
                "level": log_data["level"],
                "context": json.dumps(log_data["context"]) if log_data["context"] else None,
                "timestamp": log_data["timestamp"],
                "created_at": log_data["created_at"]
            })
            log_id = result.scalar()
            
            logger.debug(f"Created agent log {log_id} for pipeline {pipeline_run_id}")
            return str(log_id)
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to create agent log: {e}")
        return None
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize agent log context: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in agent logging: {e}")
        return None

def get_agent_logs(
    pipeline_run_id: str,
    limit: int = 100,
    offset: int = 0,
    level: Optional[str] = None,
    agent_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve agent logs for a pipeline run."""
    try:
        with get_sync_session() as session:
            base_query = """
                SELECT id, pipeline_run_id, agent_name, stage, operation, 
                       message, context, level, timestamp, created_at
                FROM agent_logs 
                WHERE pipeline_run_id = :pipeline_run_id
            """
            
            params = {"pipeline_run_id": pipeline_run_id}
            
            if level:
                base_query += " AND level = :level"
                params["level"] = level
                
            if agent_name:
                base_query += " AND agent_name = :agent_name"
                params["agent_name"] = agent_name
                
            base_query += " ORDER BY timestamp DESC LIMIT :limit OFFSET :offset"
            params["limit"] = limit
            params["offset"] = offset
            
            stmt = text(base_query)
            result = session.execute(stmt, params)
            rows = result.fetchall()
            
            logs = []
            for row in rows:
                logs.append({
                    "id": str(row[0]),
                    "pipeline_run_id": str(row[1]),
                    "agent_name": row[2],
                    "stage": row[3],
                    "operation": row[4],
                    "message": row[5],
                    "context": row[6],
                    "level": row[7],
                    "timestamp": row[8].isoformat() if row[8] else None,
                    "created_at": row[9].isoformat() if row[9] else None
                })
            
            return logs
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to get agent logs: {e}")
        return []

def delete_agent_logs(pipeline_run_id: str) -> int:
    """Delete all agent logs for a pipeline run and return count of deleted logs."""
    try:
        with get_sync_session() as session:
            stmt = text("DELETE FROM agent_logs WHERE pipeline_run_id = :pipeline_run_id")
            result = session.execute(stmt, {"pipeline_run_id": pipeline_run_id})
            deleted_count = result.rowcount
            
            logger.info(f"Deleted {deleted_count} agent logs for pipeline {pipeline_run_id}")
            return deleted_count
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to delete agent logs: {e}")
        return 0

def log_agent_activity(
    pipeline_run_id: str,
    agent_name: str,
    stage: str,
    operation: str,
    message: str,
    level: str = "INFO",
    context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Convenience function to log agent activity."""
    return create_agent_log(
        pipeline_run_id=pipeline_run_id,
        agent_name=agent_name,
        stage=stage,
        operation=operation,
        message=message,
        level=level,
        context=context
    )