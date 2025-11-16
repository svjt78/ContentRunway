import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db.sync_database import (
    get_sync_session,
    create_human_review,
    update_pipeline_status,
)
from app.models.pipeline import ContentDraft

logger = logging.getLogger(__name__)


class DraftPersistenceError(Exception):
    """Raised when the draft cannot be persisted to the database."""


class ReviewSessionError(Exception):
    """Raised when a human review session cannot be created or stored."""


def persist_content_draft(state: Dict[str, Any], max_attempts: int = 2) -> str:
    """Persist the current draft to the database and return its ID."""
    draft = state.get("draft")
    topic_id = state.get("chosen_topic_id")
    run_id = state.get("run_id")

    if not draft or not topic_id or not run_id:
        raise DraftPersistenceError("Draft persistence requires draft, topic, and run ID")

    payload = _build_draft_payload(state)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            with get_sync_session() as session:
                draft_id = uuid.uuid4()
                content_draft = ContentDraft(
                    id=draft_id,
                    pipeline_run_id=uuid.UUID(str(run_id)),
                    topic_id=uuid.UUID(str(topic_id)),
                    version=1,
                    stage="initial",
                    title=payload["title"],
                    subtitle=payload["subtitle"],
                    abstract=payload["abstract"],
                    outline=payload["outline"],
                    content=payload["content"],
                    citations=payload["citations"],
                    internal_links=payload["internal_links"],
                    word_count=payload["word_count"],
                    reading_time_minutes=payload["reading_time_minutes"],
                    readability_score=payload["readability_score"],
                    meta_description=payload["meta_description"],
                    keywords=payload["keywords"],
                    tags=payload["tags"],
                    review_status="draft",
                    is_current=False,
                )
                session.add(content_draft)
                session.commit()
                logger.info("💾 Draft persisted for pipeline %s: %s", run_id, draft_id)
                return str(draft_id)
        except SQLAlchemyError as exc:
            last_error = exc
            logger.warning(
                "Draft persistence attempt %s/%s failed for run %s: %s",
                attempt,
                max_attempts,
                run_id,
                exc,
            )

    raise DraftPersistenceError(f"Failed to persist content draft: {last_error}")


def ensure_current_draft_id(state: Dict[str, Any]) -> str:
    """Ensure state contains current_draft_id; fetch latest if missing."""
    if state.get("current_draft_id"):
        return state["current_draft_id"]

    run_id = state.get("run_id")
    if not run_id:
        raise DraftPersistenceError("Missing run_id for draft lookup")

    with get_sync_session() as session:
        query = text(
            """
            SELECT id FROM content_drafts
            WHERE pipeline_run_id = :run_id
            ORDER BY version DESC, created_at DESC
            LIMIT 1
            """
        )
        result = session.execute(query, {"run_id": str(run_id)})
        row = result.fetchone()
        if not row:
            raise DraftPersistenceError(f"No content drafts found for pipeline {run_id}")
        draft_id = str(row.id)
        state["current_draft_id"] = draft_id
        logger.info("✅ Adopted content draft %s for pipeline %s", draft_id, run_id)
        return draft_id


def create_review_session(state: Dict[str, Any], reviewer_id: str = "personal") -> str:
    """Create and store a human review session, returning the session ID."""
    draft_id = ensure_current_draft_id(state)

    review_id = create_human_review(
        pipeline_run_id=state["run_id"],
        content_draft_id=draft_id,
        reviewer_id=reviewer_id,
        status="pending",
        target_time_seconds=900,
        checklist_items={
            "technical_accuracy": False,
            "domain_expertise": False,
            "style_consistency": False,
            "compliance": False,
            "overall_quality": False,
        },
        quality_concerns=[],
        inline_edits=[],
        structural_changes=[],
    )

    if not review_id:
        raise ReviewSessionError("Failed to create human review session")

    state["human_review_session_id"] = review_id
    state["review_session_id"] = review_id

    payload = _build_review_session_payload(state, draft_id, review_id)
    if payload:
        _store_review_session_payload(review_id, payload)
        state["review_session_expires_at"] = payload.get("expires_at")
    else:
        logger.warning(
            "Review session %s created without payload (draft %s)", review_id, draft_id
        )

    # Persist reference on pipeline record for UI/CTA
    update_pipeline_status(
        state["run_id"],
        "running",
        current_step="human_review_pending",
        progress_percentage=90.0,
        review_session_id=review_id,
    )

    return review_id


def _build_draft_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize draft object into plain JSON-safe values."""
    draft = state["draft"]
    payload = {
        "title": getattr(draft, "title", "Untitled"),
        "subtitle": getattr(draft, "subtitle", None),
        "abstract": getattr(draft, "abstract", None),
        "content": getattr(draft, "content", ""),
        "word_count": getattr(draft, "word_count", 0),
        "reading_time_minutes": getattr(draft, "reading_time_minutes", 0),
        "readability_score": getattr(draft, "readability_score", None),
        "meta_description": getattr(draft, "meta_description", None),
        "keywords": _normalize_list(getattr(draft, "keywords", [])),
        "tags": _normalize_list(getattr(draft, "tags", [])),
        "citations": _serialize_citations(getattr(draft, "citations", [])),
        "internal_links": _normalize_list(getattr(draft, "internal_links", [])),
        "outline": _serialize_outline(state.get("outline")),
    }
    return payload


def _serialize_citations(citations: Any) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for citation in citations or []:
        if hasattr(citation, "__dict__"):
            serialized.append(
                {
                    "number": getattr(citation, "number", None),
                    "quote_text": getattr(citation, "quote_text", ""),
                    "context": getattr(citation, "context", ""),
                    "citation_type": getattr(citation, "citation_type", "reference"),
                    "source": _serialize_source(getattr(citation, "source", None)),
                }
            )
        else:
            serialized.append(citation)
    return serialized


def _serialize_source(source: Any) -> Dict[str, Any]:
    if source and hasattr(source, "__dict__"):
        pub_date = getattr(source, "publication_date", None)
        if pub_date and hasattr(pub_date, "isoformat"):
            pub_date = pub_date.isoformat()
        return {
            "url": getattr(source, "url", ""),
            "title": getattr(source, "title", ""),
            "author": getattr(source, "author", None),
            "publication_date": pub_date,
        }
    return source or {}


def _serialize_outline(outline: Any) -> Optional[str]:
    if not outline:
        return None
    if hasattr(outline, "dict"):
        return json.dumps(outline.dict())
    if isinstance(outline, dict):
        return json.dumps(outline)
    return json.dumps({"raw": str(outline)})


def _normalize_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value]
    return [value]


def _build_review_session_payload(
    state: Dict[str, Any], content_draft_id: str, review_id: str
) -> Optional[Dict[str, Any]]:
    content_info = _fetch_content_draft_for_review(content_draft_id)
    draft_from_state = state.get("draft")

    if not content_info and draft_from_state:
        content_info = {
            "title": getattr(draft_from_state, "title", "Untitled"),
            "subtitle": getattr(draft_from_state, "subtitle", None),
            "abstract": getattr(draft_from_state, "abstract", None),
            "content": getattr(draft_from_state, "content", ""),
            "word_count": getattr(draft_from_state, "word_count", 0),
            "reading_time_minutes": getattr(
                draft_from_state, "reading_time_minutes", 0
            ),
            "meta_description": getattr(
                draft_from_state, "meta_description", None
            ),
            "keywords": getattr(draft_from_state, "keywords", []),
            "tags": getattr(draft_from_state, "tags", []),
        }

    if not content_info:
        logger.error(
            "Unable to load content draft %s for review session %s",
            content_draft_id,
            review_id,
        )
        return None

    now = datetime.now()
    expires_at = now + timedelta(minutes=60)
    quality_scores = state.get("quality_scores") or {}

    payload = {
        "session_id": str(review_id),
        "run_id": state.get("run_id"),
        "review_session_url": f"/review/session/{review_id}",
        "status": "pending",
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "time_limit_seconds": 15 * 60,
        "review_data": {
            "quality_scores": quality_scores,
            "checklist_items": {
                "technical_accuracy": False,
                "domain_expertise": False,
                "style_consistency": False,
                "compliance": False,
                "overall_quality": False,
            },
            "domain_focus": state.get("domain_focus", []),
            "target_word_count": state.get("target_word_count"),
            "review_notes": state.get("critique_notes", []),
        },
        "content_for_review": content_info,
        "human_feedback": None,
    }
    return payload


def _fetch_content_draft_for_review(content_draft_id: str) -> Optional[Dict[str, Any]]:
    try:
        with get_sync_session() as session:
            query = text(
                """
                SELECT title, subtitle, abstract, content, word_count, reading_time_minutes,
                       meta_description, keywords, tags
                FROM content_drafts
                WHERE id = :content_id
                """
            )
            result = session.execute(query, {"content_id": content_draft_id})
            row = result.fetchone()

            if not row:
                return None

            def _parse_json_field(value):
                if value is None:
                    return []
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        return parsed if isinstance(parsed, list) else [value]
                    except json.JSONDecodeError:
                        return [value]
                return value

            return {
                "title": row.title,
                "subtitle": row.subtitle,
                "abstract": row.abstract,
                "content": row.content,
                "word_count": row.word_count,
                "reading_time_minutes": row.reading_time_minutes,
                "meta_description": row.meta_description,
                "keywords": _parse_json_field(row.keywords),
                "tags": _parse_json_field(row.tags),
            }
    except SQLAlchemyError as exc:
        logger.error("Failed to fetch content draft %s: %s", content_draft_id, exc)
        return None


def _store_review_session_payload(
    review_id: str, payload: Dict[str, Any], ttl_seconds: int = 3600
):
    try:
        import redis
        import os
        from urllib.parse import urlparse

        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        url_parts = urlparse(redis_url)
        redis_client = redis.Redis(
            host=url_parts.hostname or "redis",
            port=url_parts.port or 6379,
            db=int(url_parts.path.lstrip("/")) if url_parts.path else 0,
            decode_responses=False,
        )
        redis_client.ping()
        redis_key = f"session:{review_id}"
        serialized_payload = json.dumps(payload, default=str).encode("utf-8")
        redis_client.setex(redis_key, ttl_seconds, serialized_payload)
        redis_client.close()
        logger.info(
            "Review session payload persisted to Redis",
            extra={
                "session_id": review_id,
                "ttl_seconds": ttl_seconds,
                "redis_key": redis_key,
            },
        )
    except Exception as exc:
        logger.error("Failed to store review session payload for %s: %s", review_id, exc)
