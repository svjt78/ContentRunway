# Pipeline Persistence & Human Review Stability Plan

## Goal
Stop the repeated failures in the writing → human-review handoff by centralizing draft persistence, providing a reliable review-session creation flow, and keeping the Celery pipeline synchronous for compatibility with the existing agents.

---

## Design Overview

### 1. Draft Persistence Helper
- Create `backend/app/services/draft_persistence.py`.
- Expose `persist_content_draft(state: Dict[str, Any]) -> str`.
- Responsibilities:
  - Serialize the current draft once (title, body, outline, citations, keywords, tags, etc.).
  - Use `get_sync_session()` with ORM models to insert a `ContentDraft`.
  - Retry once or twice if the DB connection is in a failed transaction before raising `DraftPersistenceError`.
  - Return the new draft ID so downstream stages always have a valid `current_draft_id`.

### 2. Shared Review Session Helper
- Add `create_review_session(state: Dict[str, Any]) -> str` (same module or a sibling helper).
- Responsibilities:
  - Verify `state["current_draft_id"]` exists (fall back to DB lookup if needed).
  - Invoke `create_human_review`, build the Redis payload, and store it via `_store_review_session_payload`.
  - Update pipeline status with the generated `review_session_id`.
  - Raise `ReviewSessionError` on any failure so the pipeline fails explicitly instead of pausing with an empty session.

### 3. Pipeline State Snapshot Helper (Optional but recommended)
- Provide `persist_pipeline_state(state, celery_task, *, progress_step: float, current_step: str)` to wrap `_update_pipeline_progress` and `_store_pipeline_state`.
- Ensures every stage updates Redis and Celery consistently, reducing repeated code.

### 4. Keep Celery Workers Sync-Only
- Continue using synchronous helpers so editing and other agents are not blocked on async event-loop issues.
- Helpers encapsulate DB access, so each stage only needs to call the helper and react to success/failure.

### 5. Front-End Alignment
- The existing UI fix (only show “Review Content” when `review_session_id` is truthy) remains valid because the helper guarantees the ID is set before the pipeline pauses.

---

## Implementation Plan

1. **Create helper module**
   - File: `backend/app/services/draft_persistence.py`.
   - Define `DraftPersistenceError`, `ReviewSessionError`, `persist_content_draft`, `create_review_session`, and optionally `persist_pipeline_state`.
   - Include serialization of draft data (reuse `_prepare_serializable_state` logic where possible) and DB retries.

2. **Refactor writing stage**
   - Replace the inline SQL/fallback block inside `_execute_writing_stage` with:
     ```python
     from app.services.draft_persistence import persist_content_draft, DraftPersistenceError
     try:
         draft_id = persist_content_draft(state)
         state["current_draft_id"] = draft_id
     except DraftPersistenceError as exc:
         state["status"] = "failed"
         state["error_message"] = f"Writing failed: {exc}"
         return state
     ```
   - This removes duplicate code and ensures every failure is handled consistently.

3. **Refactor human-review stage**
   - Inside `_execute_human_review_stage`, after ensuring a draft exists, call `create_review_session(state)`.
   - On success, set `human_review_required = True`, store pipeline state, and pause as today.
   - On `ReviewSessionError`, fail the pipeline with the provided message.

4. **Adopt pipeline state helper (optional)**
   - Replace repeated `_update_pipeline_progress` + `_store_pipeline_state` calls with `persist_pipeline_state(...)`.
   - This reduces copy/paste bugs and makes it easier to maintain consistent progress reporting.

5. **Testing / verification**
   - Run the pipeline through writing → human-review multiple times to ensure drafts persist and review sessions are always available.
   - Confirm the CTA only appears with valid `review_session_id`.

This approach confines persistence logic to dedicated helpers, keeps Celery synchronous, and eliminates the cascading “fix one bug, introduce another” cycle that came from repeating ad-hoc DB code in multiple stages.
