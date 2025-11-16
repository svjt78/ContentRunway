from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uuid
import json

from app.db.database import get_db
from app.services.content_service import ContentService

router = APIRouter()

# Pydantic models for request/response
class ReviewContentRequest(BaseModel):
    review_notes: Optional[str] = None

class ApproveContentRequest(ReviewContentRequest):
    pass

class RejectContentRequest(ReviewContentRequest):
    required_changes: Optional[List[str]] = None

class UpdateContentRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    meta_description: Optional[str] = None
    keywords: Optional[List[str]] = None
    review_notes: Optional[str] = None

class PublishContentRequest(BaseModel):
    platforms: List[str] = ["digitaldossier"]
    category: str = "blog"  # or "product"
    cover_image_processing: bool = True

@router.get("/drafts/{run_id}")
async def get_content_drafts(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get content drafts for a pipeline run."""
    try:
        query = text("""
            SELECT 
                id, pipeline_run_id, topic_id, version, stage, title, subtitle,
                abstract, outline, content, citations, internal_links, word_count,
                reading_time_minutes, readability_score, meta_description, keywords,
                tags, review_status, reviewed_at, review_notes, published_at,
                created_at, is_current
            FROM content_drafts 
            WHERE pipeline_run_id = :run_id 
            AND (is_current = true OR stage = 'human_review_pending')
            AND stage != 'initial'
            ORDER BY version DESC, created_at DESC
        """)
        
        result = await db.execute(query, {"run_id": str(run_id)})
        rows = result.fetchall()
        
        if not rows:
            return []
        
        drafts = []
        for row in rows:
            draft = {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "topic_id": str(row.topic_id) if row.topic_id else None,
                "version": row.version,
                "stage": row.stage,
                "title": row.title,
                "subtitle": row.subtitle,
                "abstract": row.abstract,
                "outline": row.outline,
                "content": row.content,
                "citations": row.citations,
                "internal_links": row.internal_links,
                "word_count": row.word_count,
                "reading_time_minutes": row.reading_time_minutes,
                "readability_score": row.readability_score,
                "meta_description": row.meta_description,
                "keywords": row.keywords if row.keywords else [],
                "tags": row.tags if row.tags else [],
                "review_status": row.review_status,
                "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
                "review_notes": row.review_notes,
                "published_at": row.published_at.isoformat() if row.published_at else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "is_current": row.is_current
            }
            drafts.append(draft)
        
        return drafts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch content drafts: {str(e)}")

@router.get("/outlines/{run_id}")
async def get_content_outlines(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get content outlines for a pipeline run."""
    try:
        query = text("""
            SELECT 
                id, pipeline_run_id, topic_id, title, target_keywords, sections,
                meta_description, estimated_reading_time, content_structure,
                created_at
            FROM content_outlines 
            WHERE pipeline_run_id = :run_id 
            ORDER BY created_at DESC
        """)
        
        result = await db.execute(query, {"run_id": str(run_id)})
        rows = result.fetchall()
        
        if not rows:
            return []
        
        outlines = []
        for row in rows:
            outline = {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "topic_id": str(row.topic_id) if row.topic_id else None,
                "title": row.title,
                "target_keywords": row.target_keywords if row.target_keywords else [],
                "sections": row.sections if row.sections else [],
                "sections_count": len(row.sections) if row.sections else 0,
                "meta_description": row.meta_description,
                "estimated_reading_time": row.estimated_reading_time,
                "content_structure": row.content_structure,
                "created_at": row.created_at.isoformat() if row.created_at else None
            }
            outlines.append(outline)
        
        return outlines
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch content outlines: {str(e)}")

@router.get("/channels/{run_id}")
async def get_channel_content(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get channel content for a pipeline run."""
    try:
        query = text("""
            SELECT 
                id, pipeline_run_id, content_draft_id, channel, platform, title,
                formatted_content, excerpt, meta_data, publication_status,
                published_url, cover_image_url, tags, created_at
            FROM channel_content 
            WHERE pipeline_run_id = :run_id 
            ORDER BY platform, created_at DESC
        """)
        
        result = await db.execute(query, {"run_id": str(run_id)})
        rows = result.fetchall()
        
        if not rows:
            return []
        
        channel_content = []
        for row in rows:
            content = {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "content_draft_id": str(row.content_draft_id) if row.content_draft_id else None,
                "channel": row.channel,
                "platform": row.platform,
                "title": row.title,
                "formatted_content": row.formatted_content,
                "excerpt": row.excerpt,
                "meta_data": row.meta_data if row.meta_data else {},
                "publication_status": row.publication_status,
                "published_url": row.published_url,
                "cover_image_url": row.cover_image_url,
                "tags": row.tags if row.tags else [],
                "created_at": row.created_at.isoformat() if row.created_at else None
            }
            channel_content.append(content)
        
        return channel_content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch channel content: {str(e)}")

@router.get("/topics/{run_id}")
async def get_pipeline_topics(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get topic ideas for a pipeline run."""
    try:
        query = text("""
            SELECT 
                id, pipeline_run_id, title, domain, description, target_keywords,
                estimated_word_count, complexity_score, relevance_score, 
                overall_score, is_selected, metadata, created_at
            FROM topic_ideas 
            WHERE pipeline_run_id = :run_id 
            ORDER BY overall_score DESC, created_at DESC
        """)
        
        result = await db.execute(query, {"run_id": str(run_id)})
        rows = result.fetchall()
        
        if not rows:
            return []
        
        topics = []
        for row in rows:
            topic = {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "title": row.title,
                "domain": row.domain,
                "description": row.description,
                "target_keywords": row.target_keywords if row.target_keywords else [],
                "estimated_word_count": row.estimated_word_count,
                "complexity_score": row.complexity_score,
                "relevance_score": row.relevance_score,
                "overall_score": row.overall_score,
                "is_selected": row.is_selected,
                "metadata": row.metadata if row.metadata else {},
                "created_at": row.created_at.isoformat() if row.created_at else None
            }
            topics.append(topic)
        
        return topics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch topics: {str(e)}")

@router.get("/sources/{run_id}")
async def get_research_sources(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get research sources for a pipeline run."""
    try:
        query = text("""
            SELECT 
                id, pipeline_run_id, topic_id, url, title, author, publication_date,
                domain, source_type, summary, key_points, quotable_content,
                credibility_score, relevance_score, currency_score, citation_count,
                used_in_content, created_at
            FROM research_sources 
            WHERE pipeline_run_id = :run_id 
            ORDER BY relevance_score DESC, created_at DESC
        """)
        
        result = await db.execute(query, {"run_id": str(run_id)})
        rows = result.fetchall()
        
        if not rows:
            return []
        
        sources = []
        for row in rows:
            source = {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "topic_id": str(row.topic_id) if row.topic_id else None,
                "url": row.url,
                "title": row.title,
                "author": row.author,
                "publication_date": row.publication_date.isoformat() if row.publication_date else None,
                "domain": row.domain,
                "source_type": row.source_type,
                "summary": row.summary,
                "key_points": row.key_points if row.key_points else [],
                "quotable_content": row.quotable_content if row.quotable_content else [],
                "credibility_score": row.credibility_score,
                "relevance_score": row.relevance_score,
                "currency_score": row.currency_score,
                "citation_count": row.citation_count,
                "used_in_content": row.used_in_content,
                "created_at": row.created_at.isoformat() if row.created_at else None
            }
            sources.append(source)
        
        return sources
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch research sources: {str(e)}")

# New review system endpoints

@router.get("/pending-review")
async def get_pending_review_content(
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get all content pending human review."""
    try:
        content_service = ContentService(db)
        return await content_service.get_pending_review_content()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch pending content: {str(e)}")

@router.get("/{content_id}")
async def get_content_by_id(
    content_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get content draft by ID."""
    try:
        content_service = ContentService(db)
        content = await content_service.get_content_by_id(str(content_id))
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        return content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch content: {str(e)}")

@router.post("/{content_id}/approve")
async def approve_content(
    content_id: uuid.UUID,
    request: ApproveContentRequest,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Approve content for publishing."""
    try:
        content_service = ContentService(db)
        
        # Update content status to approved
        result = await content_service.update_content_review_status(
            content_id=str(content_id),
            status="approved",
            review_notes=request.review_notes,
            reviewer_id="current_user"  # TODO: Get from auth context
        )
        
        # Trigger publishing pipeline if needed
        publishing_result = await content_service.trigger_publishing_for_approved_content(str(content_id))
        
        return {
            "success": True,
            "content_id": result["id"],
            "review_status": result["review_status"],
            "reviewed_at": result["reviewed_at"].isoformat() if result["reviewed_at"] else None,
            "publishing_started": publishing_result.get("success", False),
            "publishing_status": publishing_result.get("status", "unknown"),
            "message": "Content approved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve content: {str(e)}")

@router.post("/{content_id}/reject")
async def reject_content(
    content_id: uuid.UUID,
    request: RejectContentRequest,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Reject content with feedback."""
    try:
        content_service = ContentService(db)
        
        # Prepare review notes with required changes
        review_notes = request.review_notes or ""
        if request.required_changes:
            changes_text = "\n".join([f"- {change}" for change in request.required_changes])
            review_notes = f"{review_notes}\n\nRequired changes:\n{changes_text}".strip()
        
        result = await content_service.update_content_review_status(
            content_id=str(content_id),
            status="rejected",
            review_notes=review_notes,
            reviewer_id="current_user"  # TODO: Get from auth context
        )
        
        return {
            "success": True,
            "content_id": result["id"],
            "review_status": result["review_status"],
            "reviewed_at": result["reviewed_at"].isoformat() if result["reviewed_at"] else None,
            "message": "Content rejected with feedback"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reject content: {str(e)}")

@router.put("/{content_id}")
async def update_content(
    content_id: uuid.UUID,
    request: UpdateContentRequest,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Update content data and reset to draft status."""
    try:
        content_service = ContentService(db)
        
        result = await content_service.update_content_data(
            content_id=str(content_id),
            title=request.title,
            content=request.content,
            meta_description=request.meta_description,
            keywords=request.keywords,
            review_notes=request.review_notes
        )
        
        return {
            "success": True,
            "content_id": result["id"],
            "title": result["title"],
            "review_status": result["review_status"],
            "word_count": result["word_count"],
            "message": "Content updated successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update content: {str(e)}")

@router.delete("/{content_id}")
async def delete_content(
    content_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Delete content draft."""
    try:
        content_service = ContentService(db)
        
        success = await content_service.delete_content(str(content_id))
        if not success:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {
            "success": True,
            "content_id": str(content_id),
            "message": "Content deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete content: {str(e)}")

@router.post("/{content_id}/publish")
async def trigger_publishing(
    content_id: uuid.UUID,
    request: PublishContentRequest,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Trigger publishing for approved content."""
    try:
        content_service = ContentService(db)
        
        # Get content to verify it's approved
        content = await content_service.get_content_by_id(str(content_id))
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        if content["review_status"] != "approved":
            raise HTTPException(
                status_code=400, 
                detail=f"Content must be approved before publishing. Current status: {content['review_status']}"
            )
        
        # TODO: Trigger publishing pipeline with Celery
        # For now, just return a placeholder response
        
        return {
            "success": True,
            "content_id": str(content_id),
            "publishing_job_id": str(uuid.uuid4()),  # Placeholder
            "status": "publishing",
            "message": "Publishing job started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger publishing: {str(e)}")