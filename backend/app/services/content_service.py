"""
Content service for persisting pipeline-generated content to database.
"""

from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import uuid
from datetime import datetime
import json
from typing import Any


class ContentService:
    """Service for managing content drafts in the database."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def save_content_draft(
        self,
        pipeline_run_id: str,
        topic_id: str,
        draft_content: Dict[str, Any],
        stage: str = "initial",
        version: int = 1
    ) -> str:
        """
        Save a content draft to the database.
        
        Args:
            pipeline_run_id: ID of the pipeline run
            topic_id: ID of the selected topic
            draft_content: Content draft data from pipeline state
            stage: Stage of the content (initial, edited, final)
            version: Version number of the draft
            
        Returns:
            ID of the created content draft
        """
        try:
            # Extract content information from draft
            title = draft_content.get("title", "Untitled")
            subtitle = draft_content.get("subtitle")
            abstract = draft_content.get("abstract")
            content = draft_content.get("content", "")
            word_count = draft_content.get("word_count", len(content.split()) if content else 0)
            reading_time = draft_content.get("reading_time_minutes", max(1, word_count // 200))
            readability_score = draft_content.get("readability_score")
            meta_description = draft_content.get("meta_description")
            keywords = draft_content.get("keywords", [])
            tags = draft_content.get("tags", [])
            citations = draft_content.get("citations", [])
            internal_links = draft_content.get("internal_links", [])
            outline = draft_content.get("outline")
            
            # Generate new ID
            draft_id = str(uuid.uuid4())
            
            # Prepare outline as JSON
            outline_json = None
            if outline:
                if hasattr(outline, 'dict'):
                    outline_json = json.dumps(outline.dict())
                elif isinstance(outline, dict):
                    outline_json = json.dumps(outline)
                else:
                    outline_json = json.dumps({"raw": str(outline)})
            
            # Insert content draft
            query = text("""
                INSERT INTO content_drafts (
                    id, pipeline_run_id, topic_id, version, stage, title, subtitle,
                    abstract, outline, content, citations, internal_links, word_count,
                    reading_time_minutes, readability_score, meta_description, keywords,
                    tags, review_status, created_at, is_current
                ) VALUES (
                    :id, :pipeline_run_id, :topic_id, :version, :stage, :title, :subtitle,
                    :abstract, :outline, :content, :citations, :internal_links, :word_count,
                    :reading_time_minutes, :readability_score, :meta_description, :keywords,
                    :tags, :review_status, NOW(), :is_current
                )
            """)
            
            await self.db.execute(query, {
                "id": draft_id,
                "pipeline_run_id": pipeline_run_id,
                "topic_id": topic_id,
                "version": version,
                "stage": stage,
                "title": title,
                "subtitle": subtitle,
                "abstract": abstract,
                "outline": outline_json,
                "content": content,
                "citations": json.dumps(citations),
                "internal_links": json.dumps(internal_links),
                "word_count": word_count,
                "reading_time_minutes": reading_time,
                "readability_score": readability_score,
                "meta_description": meta_description,
                "keywords": json.dumps(keywords),
                "tags": json.dumps(tags),
                "review_status": "draft",  # Default status for new content
                "is_current": stage == "final"  # Only mark final drafts as current
            })
            
            await self.db.commit()
            
            return draft_id
            
        except Exception as e:
            await self.db.rollback()
            raise Exception(f"Failed to save content draft: {str(e)}")
    
    async def update_content_draft(
        self,
        draft_id: str,
        updated_content: Dict[str, Any],
        stage: str,
        increment_version: bool = True
    ) -> str:
        """
        Update an existing content draft.
        
        Args:
            draft_id: ID of the draft to update
            updated_content: Updated content data
            stage: New stage of the content
            increment_version: Whether to increment version number
            
        Returns:
            ID of the updated draft (may be new ID if version incremented)
        """
        try:
            if increment_version:
                # Create new version instead of updating existing
                # First get the current draft info
                query = text("""
                    SELECT pipeline_run_id, topic_id, version 
                    FROM content_drafts 
                    WHERE id = :draft_id
                """)
                result = await self.db.execute(query, {"draft_id": draft_id})
                row = result.fetchone()
                
                if not row:
                    raise ValueError(f"Draft with ID {draft_id} not found")
                
                # Create new version
                new_version = row.version + 1
                return await self.save_content_draft(
                    pipeline_run_id=str(row.pipeline_run_id),
                    topic_id=str(row.topic_id),
                    draft_content=updated_content,
                    stage=stage,
                    version=new_version
                )
            else:
                # Update existing draft in place
                title = updated_content.get("title", "Untitled")
                content = updated_content.get("content", "")
                word_count = updated_content.get("word_count", len(content.split()) if content else 0)
                reading_time = updated_content.get("reading_time_minutes", max(1, word_count // 200))
                readability_score = updated_content.get("readability_score")
                
                query = text("""
                    UPDATE content_drafts SET 
                        stage = :stage,
                        title = :title,
                        content = :content,
                        word_count = :word_count,
                        reading_time_minutes = :reading_time_minutes,
                        readability_score = :readability_score,
                        is_current = :is_current
                    WHERE id = :draft_id
                """)
                
                await self.db.execute(query, {
                    "draft_id": draft_id,
                    "stage": stage,
                    "title": title,
                    "content": content,
                    "word_count": word_count,
                    "reading_time_minutes": reading_time,
                    "readability_score": readability_score,
                    "is_current": stage == "final"
                })
                
                await self.db.commit()
                return draft_id
                
        except Exception as e:
            await self.db.rollback()
            raise Exception(f"Failed to update content draft: {str(e)}")
    
    async def get_latest_draft_id(self, pipeline_run_id: str) -> Optional[str]:
        """Get the ID of the latest content draft for a pipeline run."""
        try:
            query = text("""
                SELECT id FROM content_drafts 
                WHERE pipeline_run_id = :pipeline_run_id 
                ORDER BY version DESC, created_at DESC 
                LIMIT 1
            """)
            result = await self.db.execute(query, {"pipeline_run_id": pipeline_run_id})
            row = result.fetchone()
            
            return str(row.id) if row else None
            
        except Exception as e:
            raise Exception(f"Failed to get latest draft ID: {str(e)}")
    
    async def mark_draft_as_current(self, draft_id: str) -> None:
        """Mark a specific draft as the current one and unmark others."""
        try:
            # First get the pipeline run ID
            query = text("SELECT pipeline_run_id FROM content_drafts WHERE id = :draft_id")
            result = await self.db.execute(query, {"draft_id": draft_id})
            row = result.fetchone()
            
            if not row:
                raise ValueError(f"Draft with ID {draft_id} not found")
            
            pipeline_run_id = row.pipeline_run_id
            
            # Unmark all drafts for this pipeline run
            query = text("""
                UPDATE content_drafts SET is_current = FALSE 
                WHERE pipeline_run_id = :pipeline_run_id
            """)
            await self.db.execute(query, {"pipeline_run_id": str(pipeline_run_id)})
            
            # Mark the specified draft as current
            query = text("""
                UPDATE content_drafts SET is_current = TRUE 
                WHERE id = :draft_id
            """)
            await self.db.execute(query, {"draft_id": draft_id})
            
            await self.db.commit()
            
        except Exception as e:
            await self.db.rollback()
            raise Exception(f"Failed to mark draft as current: {str(e)}")
    
    async def update_content_review_status(
        self,
        content_id: str,
        status: str,
        review_notes: Optional[str] = None,
        reviewer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update content review status and return updated content."""
        try:
            # Validate status
            valid_statuses = ['draft', 'approved', 'rejected', 'published']
            if status not in valid_statuses:
                raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
            
            query = text("""
                UPDATE content_drafts SET 
                    review_status = CAST(:status AS VARCHAR),
                    reviewed_at = CASE WHEN CAST(:status AS VARCHAR) = 'draft' THEN NULL ELSE NOW() END,
                    review_notes = :review_notes,
                    reviewer_id = :reviewer_id,
                    published_at = CASE WHEN CAST(:status AS VARCHAR) = 'published' THEN NOW() ELSE published_at END
                WHERE id = :content_id
                RETURNING id, title, review_status, reviewed_at, pipeline_run_id
            """)
            
            result = await self.db.execute(query, {
                "content_id": content_id,
                "status": status,
                "review_notes": review_notes,
                "reviewer_id": reviewer_id
            })
            
            row = result.fetchone()
            if not row:
                raise ValueError(f"Content with ID {content_id} not found")
            
            await self.db.commit()
            
            return {
                "id": str(row.id),
                "title": row.title,
                "review_status": row.review_status,
                "reviewed_at": row.reviewed_at,
                "pipeline_run_id": str(row.pipeline_run_id)
            }
            
        except Exception as e:
            await self.db.rollback()
            raise Exception(f"Failed to update review status: {str(e)}")
    
    async def get_content_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get content draft by ID with all fields."""
        try:
            query = text("""
                SELECT 
                    id, pipeline_run_id, topic_id, version, stage, title, subtitle,
                    abstract, outline, content, citations, internal_links, word_count,
                    reading_time_minutes, readability_score, meta_description, keywords,
                    tags, review_status, reviewed_at, review_notes, reviewer_id,
                    published_at, published_urls, created_at, is_current
                FROM content_drafts 
                WHERE id = :content_id
            """)
            
            result = await self.db.execute(query, {"content_id": content_id})
            row = result.fetchone()
            
            if not row:
                return None
            
            return {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "topic_id": str(row.topic_id),
                "version": row.version,
                "stage": row.stage,
                "title": row.title,
                "subtitle": row.subtitle,
                "abstract": row.abstract,
                "outline": _safe_json_load(row.outline),
                "content": row.content,
                "citations": _safe_json_load(row.citations) or [],
                "internal_links": _safe_json_load(row.internal_links) or [],
                "word_count": row.word_count,
                "reading_time_minutes": row.reading_time_minutes,
                "readability_score": row.readability_score,
                "meta_description": row.meta_description,
                "keywords": _safe_json_load(row.keywords) or [],
                "tags": _safe_json_load(row.tags) or [],
                "review_status": row.review_status,
                "reviewed_at": row.reviewed_at,
                "review_notes": row.review_notes,
                "reviewer_id": row.reviewer_id,
                "published_at": row.published_at,
                "published_urls": _safe_json_load(row.published_urls),
                "created_at": row.created_at,
                "is_current": row.is_current
            }
            
        except Exception as e:
            raise Exception(f"Failed to get content: {str(e)}")
    
    async def get_content_by_pipeline_run(self, pipeline_run_id: str) -> List[Dict[str, Any]]:
        """Get all content drafts for a pipeline run."""
        try:
            query = text("""
                SELECT 
                    id, pipeline_run_id, topic_id, version, stage, title, subtitle,
                    abstract, content, word_count, reading_time_minutes, meta_description,
                    keywords, review_status, reviewed_at, review_notes, published_at,
                    created_at, is_current
                FROM content_drafts 
                WHERE pipeline_run_id = :pipeline_run_id
                ORDER BY version DESC, created_at DESC
            """)
            
            result = await self.db.execute(query, {"pipeline_run_id": pipeline_run_id})
            rows = result.fetchall()
            
            content_list = []
            for row in rows:
                content_list.append({
                    "id": str(row.id),
                    "pipeline_run_id": str(row.pipeline_run_id),
                    "topic_id": str(row.topic_id),
                    "version": row.version,
                    "stage": row.stage,
                    "title": row.title,
                    "subtitle": row.subtitle,
                    "abstract": row.abstract,
                    "content": row.content,
                    "word_count": row.word_count,
                    "reading_time_minutes": row.reading_time_minutes,
                    "meta_description": row.meta_description,
                    "keywords": json.loads(row.keywords) if row.keywords and isinstance(row.keywords, str) else (row.keywords if row.keywords else []),
                    "review_status": row.review_status,
                    "reviewed_at": row.reviewed_at,
                    "review_notes": row.review_notes,
                    "published_at": row.published_at,
                    "created_at": row.created_at,
                    "is_current": row.is_current
                })
            
            return content_list
            
        except Exception as e:
            raise Exception(f"Failed to get content by pipeline run: {str(e)}")
    
    async def get_pending_review_content(self) -> List[Dict[str, Any]]:
        """Get all content with draft status (pending review)."""
        try:
            query = text("""
                SELECT 
                    cd.id, cd.pipeline_run_id, cd.title, cd.abstract, cd.word_count,
                    cd.reading_time_minutes, cd.created_at, cd.stage, cd.is_current,
                    pr.domain_focus
                FROM content_drafts cd
                JOIN pipeline_runs pr ON cd.pipeline_run_id = pr.id
                WHERE cd.review_status = 'draft' AND cd.is_current = true
                ORDER BY cd.created_at DESC
            """)
            
            result = await self.db.execute(query)
            rows = result.fetchall()
            
            pending_list = []
            for row in rows:
                pending_list.append({
                    "id": str(row.id),
                    "pipeline_run_id": str(row.pipeline_run_id),
                    "title": row.title,
                    "abstract": row.abstract,
                    "word_count": row.word_count,
                    "reading_time_minutes": row.reading_time_minutes,
                    "created_at": row.created_at,
                    "stage": row.stage,
                    "is_current": row.is_current,
                    "domain": json.loads(row.domain_focus) if row.domain_focus else []
                })
            
            return pending_list
            
        except Exception as e:
            raise Exception(f"Failed to get pending review content: {str(e)}")
    
    async def update_content_data(
        self,
        content_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        meta_description: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        review_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update content data and reset status to draft."""
        try:
            # Build dynamic update query
            update_fields = []
            params = {"content_id": content_id}
            
            if title is not None:
                update_fields.append("title = :title")
                params["title"] = title
                
            if content is not None:
                update_fields.append("content = :content")
                params["content"] = content
                # Update word count and reading time
                word_count = len(content.split()) if content else 0
                update_fields.append("word_count = :word_count")
                update_fields.append("reading_time_minutes = :reading_time")
                params["word_count"] = word_count
                params["reading_time"] = max(1, word_count // 200)
                
            if meta_description is not None:
                update_fields.append("meta_description = :meta_description")
                params["meta_description"] = meta_description
                
            if keywords is not None:
                update_fields.append("keywords = :keywords")
                params["keywords"] = json.dumps(keywords)
                
            if review_notes is not None:
                update_fields.append("review_notes = :review_notes")
                params["review_notes"] = review_notes
            
            # Always reset to draft status when content is edited
            update_fields.extend([
                "review_status = 'draft'",
                "reviewed_at = NULL",
                "reviewer_id = NULL"
            ])
            
            if not update_fields:
                raise ValueError("No fields to update")
            
            query = text(f"""
                UPDATE content_drafts SET 
                    {', '.join(update_fields)}
                WHERE id = :content_id
                RETURNING id, title, review_status, word_count
            """)
            
            result = await self.db.execute(query, params)
            row = result.fetchone()
            
            if not row:
                raise ValueError(f"Content with ID {content_id} not found")
            
            await self.db.commit()
            
            return {
                "id": str(row.id),
                "title": row.title,
                "review_status": row.review_status,
                "word_count": row.word_count
            }
            
        except Exception as e:
            await self.db.rollback()
            raise Exception(f"Failed to update content data: {str(e)}")
    
    async def delete_content(self, content_id: str) -> bool:
        """Delete content draft."""
        try:
            query = text("DELETE FROM content_drafts WHERE id = :content_id")
            result = await self.db.execute(query, {"content_id": content_id})
            await self.db.commit()
            
            return result.rowcount > 0
            
        except Exception as e:
            await self.db.rollback()
            raise Exception(f"Failed to delete content: {str(e)}")
    
    async def trigger_publishing_for_approved_content(self, content_id: str) -> Dict[str, Any]:
        """Check if content is approved and trigger publishing if needed."""
        try:
            # Get content to check status
            content = await self.get_content_by_id(content_id)
            if not content:
                raise ValueError(f"Content with ID {content_id} not found")
            
            if content["review_status"] != "approved":
                raise ValueError(f"Content must be approved to trigger publishing. Current status: {content['review_status']}")
            
            # Get pipeline run to check if it's waiting for publishing
            pipeline_run_id = content["pipeline_run_id"]
            pipeline_query = text("SELECT current_step, status FROM pipeline_runs WHERE id = :pipeline_id")
            result = await self.db.execute(pipeline_query, {"pipeline_id": pipeline_run_id})
            pipeline_row = result.fetchone()
            
            if pipeline_row and pipeline_row.current_step == "human_review_pending" and pipeline_row.status == "paused":
                # Import and trigger pipeline resume task
                from app.tasks.pipeline_tasks import resume_pipeline_from_publishing
                
                # Trigger pipeline resume in background
                task = resume_pipeline_from_publishing.delay(str(pipeline_run_id), content_id)
                
                return {
                    "success": True,
                    "content_id": content_id,
                    "pipeline_run_id": str(pipeline_run_id),
                    "publishing_task_id": task.id,
                    "status": "publishing_triggered",
                    "message": "Publishing pipeline resumed for approved content"
                }
            else:
                return {
                    "success": False,
                    "content_id": content_id,
                    "status": "no_action_needed",
                    "message": f"Pipeline not waiting for approval. Current step: {pipeline_row.current_step if pipeline_row else 'unknown'}, status: {pipeline_row.status if pipeline_row else 'unknown'}"
                }
            
        except Exception as e:
            raise Exception(f"Failed to trigger publishing: {str(e)}")
def _safe_json_load(value):
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
