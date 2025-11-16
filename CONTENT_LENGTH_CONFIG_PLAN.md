# Content Length Configuration Plan

This document explains how to add a user-configurable content-length control (default **500 characters**) to the ContentRunway pipeline so that the Celery agents honor the requested size end-to-end.

---

## 1. Frontend (Next.js)

**Form updates (`frontend/src/components/pipeline/StartPipelineForm.tsx`)**
- Add a numeric “Target Length (characters)” input next to the quality thresholds. Pre-populate it with 500, enforce sensible bounds (e.g., 100–4000) to stay above publisher minimums.
- Extend `PipelineFormData` to include `target_character_count` (or similar), register the control with `react-hook-form`, and include validation error messaging.
- Ensure `onSubmit` attaches `target_character_count` to the payload that is sent to `startPipeline`.

**API typings (`frontend/src/lib/api.ts`)**
- Extend `StartPipelineRequest` and `PipelineRun` interfaces to include the new field so TypeScript enforces usage everywhere the run object is displayed.
- Bubble the value wherever runs are listed (RecentRuns, PipelineOverview) so operators can confirm the requested length.

---

## 2. Backend API Layer

**Schemas (`backend/app/schemas/pipeline.py`)**
- Add `target_character_count: int = Field(default=500, ge=100)` to `PipelineRunCreate`.
- Include the field in `PipelineRunResponse` so the UI can read it back.

**Models & migrations**
- Add an `Integer` column (e.g., `target_character_count`) to `pipeline_runs` in `backend/app/models/pipeline.py`.
- Create an Alembic migration that adds the column with default 500 for existing rows.

**Service layer (`backend/app/services/pipeline_service.py`)**
- Persist the field when creating `PipelineRun` records, include it in `pipeline_config`, and copy it into the Redis state.
- Make sure `get_pipeline_run` and `list_pipeline_runs` include the new value in their response dicts.

---

## 3. Pipeline Execution (`backend/app/tasks/pipeline_tasks.py`)

1. **State initialization**  
   - Store both `target_character_count` and its derived `content_word_count_target` (character target ÷ average characters per word, clamped to reasonable limits).  
   - Persist these values in Redis via `_store_pipeline_state`.

2. **Writer validation**  
   - Replace the hard-coded `word_count < 500` failure with a tolerance check relative to `target_character_count`. Example: reject drafts outside ±15% of the requested size, or trigger editing retries that shrink/expand the content until it fits.

3. **Stage propagation**  
   - Pass the new settings into every stage invocation. The SEO stage already reads `state.get('content_word_count_target', 1500)`; once the state is populated, no code change is needed there.  
   - Editing, formatting, human review, and publishing stages must reference the target so they do not expand content back to long-form defaults.

---

## 4. Agents (langgraph/contentrunway)

### State definition
Update `ContentPipelineState` (TypedDict & Pydantic models) to add the two new fields so validations pass when Celery or agents access them.

### SEOStrategistAgent (`agents/seo.py`)
- In `_create_fallback_outline`, size section `estimated_words` so they sum to `state['content_word_count_target']` instead of the baked-in 1500-word template.

### ContentWriterAgent (`agents/writing.py`)
- Update the global instructions (currently “Generate 1200-1800 word…” in `writer_agent_role_and_goal`) to honor the dynamic target.
- While assembling sections, cap or pad output so the concatenated content approximates the requested character count. Use the derived per-section targets to scale citation/term requirements down for very short drafts.
- When `_assemble_draft` finishes, compute both word count and character count, trimming any overflow beyond the tolerance window.

### Editing Agent (`agents/editing.py`)
- Adjust optimization routines to compare current length vs. requested target; edits should focus on condensing or expanding toward the configured size instead of always increasing detail.

### Formatting Agent (`agents/formatting.py`)
- Replace the static per-platform `ideal_word_count`/`max_word_count` values with versions derived from the requested length so the formatter doesn’t lengthen the content again.

### Publisher Agent (`agents/publisher.py`)
- Ensure the publishing validation logic treats “short-form” runs as valid provided they exceed the absolute minimum (100 characters). Log the requested target when raising any “too short” errors to simplify debugging.

---

## 5. Quality Gates (`agents/quality_gates.py`)

Current scoring assumes 500+ words and “5 citations” regardless of draft size. Update the following:

- Pass `target_character_count` (or derived word count) into each gate’s `execute` call.
- In `_verify_content_readiness`, scale baseline expectations (citations, domain terms, sentence count) relative to the requested size so short-form drafts can pass without artificial failures.
- Adjust normalization factors like `length_normalization = min(1.0, words / 500)` to use the dynamic target.
- Bump `QUALITY_GATE_SCORING_VERSION` after making these changes to invalidate cached scores.

---

## 6. Documentation & Tests

- Update README, AGENTS.md, and any other developer docs to describe the new UI control and pipeline behavior.
- Modify integration/unit tests (e.g., `test_pipeline.py`, `test_full_pipeline.py`) to include the new field in `PipelineRunCreate` payloads. Add regression tests that request a short target and assert the resulting draft length falls within tolerance.
- Ensure any scripts or fixtures that instantiate `PipelineRun` records (data seeding, analytics tools) set a default for the new column.

---

## Implementation Notes

- Character-to-word conversion: `target_words = clamp(round(chars / 5), 80, 1500)` provides a stable estimate for sections and agents that still work in words.
- Tolerance window: 15–20% around the requested length typically balances flexibility with user expectations. Surface the tolerance in logs to explain auto-retries.
- Backward compatibility: default all missing values to 500 characters to keep existing automations running until the UI is redeployed.

Following the steps above will let users request short-form documents directly from the dashboard while guaranteeing every agent respects the configured length throughout the Celery pipeline.
