from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
import uuid
import logging

from app.db.database import get_db
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class ReviewFeedback(BaseModel):
    decision: str  # "approved", "rejected", "needs_revision"
    feedback_notes: str = ""
    overall_rating: int = None
    quality_concerns: List[str] = []


@router.get("/by-run/{run_id}")
async def get_human_reviews_by_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get human reviews for a pipeline run."""
    try:
        from sqlalchemy import text
        
        query = text("""
            SELECT 
                id, pipeline_run_id, content_draft_id, reviewer_id, 
                review_session_url, status, decision, started_at, completed_at,
                time_spent_seconds, target_time_seconds, overall_rating,
                feedback_notes, checklist_items, quality_concerns,
                inline_edits, structural_changes, ai_recommendations_shown,
                ai_recommendations_accepted, created_at
            FROM human_reviews 
            WHERE pipeline_run_id = :run_id 
            ORDER BY created_at DESC
        """)
        
        result = await db.execute(query, {"run_id": str(run_id)})
        rows = result.fetchall()
        
        if not rows:
            return []
        
        reviews = []
        for row in rows:
            review = {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "content_draft_id": str(row.content_draft_id) if row.content_draft_id else None,
                "reviewer_id": row.reviewer_id,
                "review_session_url": row.review_session_url,
                "status": row.status,
                "decision": row.decision,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                "time_spent_seconds": row.time_spent_seconds,
                "target_time_seconds": row.target_time_seconds,
                "overall_rating": row.overall_rating,
                "feedback_notes": row.feedback_notes,
                "checklist_items": row.checklist_items if row.checklist_items else {},
                "quality_concerns": row.quality_concerns if row.quality_concerns else [],
                "inline_edits": row.inline_edits if row.inline_edits else [],
                "structural_changes": row.structural_changes if row.structural_changes else [],
                "ai_recommendations_shown": row.ai_recommendations_shown if row.ai_recommendations_shown else [],
                "ai_recommendations_accepted": row.ai_recommendations_accepted if row.ai_recommendations_accepted else [],
                "created_at": row.created_at.isoformat() if row.created_at else None
            }
            reviews.append(review)
        
        return reviews
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch human reviews: {str(e)}")

@router.get("/session/{session_id}")
async def get_review_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get review session details for human reviewer."""
    try:
        from app.services.redis_service import get_redis_service
        import asyncio
        
        logger.info(f"Fetching review session {session_id}")
        
        # Get session data from Redis using proper context manager
        async with get_redis_service() as redis_service:
            session_data = await redis_service.get("session", session_id)
        
        if not session_data:
            logger.warning(f"Review session {session_id} not found in Redis")
            raise HTTPException(status_code=404, detail="Review session not found or expired")
        
        logger.info(
            "Review session payload loaded",
            extra={
                "session_id": session_id,
                "run_id": session_data.get("run_id"),
                "status": session_data.get("status"),
                "has_content": bool(session_data.get("content_for_review"))
            }
        )
        
        return {
            "session_id": session_id,
            "status": session_data.get("status", "pending"),
            "run_id": session_data.get("run_id"),
            "created_at": session_data.get("created_at"),
            "expires_at": session_data.get("expires_at"),
            "review_data": session_data.get("review_data", {}),
            "content_for_review": session_data.get("content_for_review", {}),
            "has_feedback": bool(session_data.get("human_feedback"))
        }
    except Exception as e:
        logger.error(f"Error retrieving review session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve review session")


@router.post("/session/{session_id}/approve")
async def approve_review(
    session_id: str,
    feedback: ReviewFeedback,
    db: AsyncSession = Depends(get_db)
):
    """Approve content in review session."""
    return await _submit_review_feedback(session_id, "approved", feedback)


@router.post("/session/{session_id}/reject")
async def reject_review(
    session_id: str,
    feedback: ReviewFeedback,
    db: AsyncSession = Depends(get_db)
):
    """Reject content in review session."""
    return await _submit_review_feedback(session_id, "rejected", feedback)


@router.post("/session/{session_id}/revise")
async def request_revision(
    session_id: str,
    feedback: ReviewFeedback,
    db: AsyncSession = Depends(get_db)
):
    """Request revision for content in review session."""
    return await _submit_review_feedback(session_id, "needs_revision", feedback)


async def _submit_review_feedback(session_id: str, decision: str, feedback: ReviewFeedback) -> Dict[str, Any]:
    """Submit human review feedback to Redis for pipeline consumption."""
    try:
        from app.services.redis_service import get_redis_service
        from datetime import datetime
        
        # Get session data from Redis using proper context manager
        async with get_redis_service() as redis_service:
            session_data = await redis_service.get("session", session_id)
            
            if not session_data:
                raise HTTPException(status_code=404, detail="Review session not found or expired")
            
            if session_data.get("human_feedback"):
                raise HTTPException(status_code=409, detail="Review feedback already submitted")
            
            # Create feedback object
            human_feedback = {
                "decision": decision,
                "feedback_notes": feedback.feedback_notes,
                "overall_rating": feedback.overall_rating,
                "quality_concerns": feedback.quality_concerns,
                "submitted_at": datetime.now().isoformat(),
                "timeout": False
            }
            
            # Update session with feedback
            session_data["human_feedback"] = human_feedback
            session_data["status"] = "completed"
            
            # Store back to Redis with extended expiry for audit
            await redis_service.set("session", session_id, session_data, ttl=24*60*60)  # 24 hours
        
        logger.info(f"Review feedback submitted for session {session_id}: {decision}")
        
        return {
            "message": f"Review {decision} successfully",
            "session_id": session_id,
            "decision": decision,
            "run_id": session_data.get("run_id"),
            "submitted_at": human_feedback["submitted_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting review feedback for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit review feedback")
