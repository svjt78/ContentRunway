from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import uuid

from app.db.database import get_db

router = APIRouter()

@router.get("/publications/{run_id}")
async def get_publications(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get publications for a pipeline run."""
    try:
        from sqlalchemy import text
        from typing import Dict, Any
        
        # Get publications with joined channel content
        query = text("""
            SELECT 
                p.id, p.pipeline_run_id, p.channel_content_id, p.platform,
                p.platform_content_id, p.published_url, p.canonical_url,
                p.title, p.slug, p.description, p.tags, p.publication_status,
                p.published_at, p.metrics_data, p.created_at,
                cc.channel, cc.formatted_content, cc.excerpt, cc.meta_data,
                cc.cover_image_url
            FROM publications p
            LEFT JOIN channel_content cc ON p.channel_content_id = cc.id
            WHERE p.pipeline_run_id = :run_id 
            ORDER BY p.published_at DESC, p.created_at DESC
        """)
        
        result = await db.execute(query, {"run_id": str(run_id)})
        rows = result.fetchall()
        
        if not rows:
            return []
        
        publications = []
        for row in rows:
            publication = {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "channel_content_id": str(row.channel_content_id) if row.channel_content_id else None,
                "platform": row.platform,
                "platform_content_id": row.platform_content_id,
                "published_url": row.published_url,
                "canonical_url": row.canonical_url,
                "title": row.title,
                "slug": row.slug,
                "description": row.description,
                "tags": row.tags if row.tags else [],
                "publication_status": row.publication_status,
                "published_at": row.published_at.isoformat() if row.published_at else None,
                "metrics_data": row.metrics_data if row.metrics_data else {},
                "created_at": row.created_at.isoformat() if row.created_at else None,
                # Channel content details
                "channel": row.channel,
                "formatted_content": row.formatted_content,
                "excerpt": row.excerpt,
                "meta_data": row.meta_data if row.meta_data else {},
                "cover_image_url": row.cover_image_url
            }
            publications.append(publication)
        
        return publications
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch publications: {str(e)}")

@router.post("/publish/{run_id}")
async def publish_content(
    run_id: uuid.UUID,
    publishing_config: dict,
    db: AsyncSession = Depends(get_db)
):
    """Publish content to configured platforms."""
    # Placeholder implementation
    return {"message": "Publishing endpoint - implementation pending"}