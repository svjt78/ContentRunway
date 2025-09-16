from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Dict, Any
import uuid
import json

from app.db.database import get_db

router = APIRouter()

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
                tags, created_at, is_current
            FROM content_drafts 
            WHERE pipeline_run_id = :run_id 
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
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "is_current": row.is_current
            }
            drafts.append(draft)
        
        return drafts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch content drafts: {str(e)}")

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