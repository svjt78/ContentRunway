# Agent Log Enhancements: Pointer + Lazy Fetch Design

## Summary

This document describes the improvement plan for the Agent Log feature so that users see meaningful outputs from each agent step without overloading the log system. The approach preserves small, performant log entries while enabling rich, on‑demand details via API lookups.

Core idea: agent logs store pointers (IDs and counts) to detailed records written to Postgres, and the UI fetches those details only when the user expands a log entry. For long content (writing/editing), the log links to the Content tab instead of embedding large bodies.

Status: Proposal ready to implement (no schema changes required)

---

## Goals

- Show “real output” for each agent step with minimal payloads in the `agent_logs` table.
- Keep Agent Log entries responsive during auto‑refresh (5s) by avoiding large JSON blobs.
- Provide navigation links to relevant content or detail views (e.g., Content tab, outline view, assessments view).
- Avoid schema changes by using existing tables and adding read‑only endpoints.

Non‑Goals
- Change existing persistence flows for agent outputs (already persisted to tables).
- Store megabyte‑sized text bodies in `agent_logs.context`.

---

## Approach

Agent Log entries remain small but include standardized pointer fields to records in Postgres (e.g., `content_draft_id`, `outline_id`, `quality_assessment_ids`). The frontend detects these pointers and, on expansion, fetches full details via dedicated GET endpoints. For writing/editing bodies, the log entry links to the Content tab filtered to the specific draft.

Example log context (indicative):

```json
{
  "topics_count": 9,
  "sources_count": 12,
  "topic_ids": ["…", "…", "…"],
  "source_ids": ["…", "…", "…"]
}
```

```json
{
  "content_draft_id": "…",
  "version": 3,
  "stage": "final"
}
```

```json
{
  "outline_id": "…",
  "sections_count": 7
}
```

```json
{
  "quality_assessment_ids": {
    "fact_check": ["…"],
    "domain_expertise": ["…"],
    "style_consistency": ["…"],
    "compliance": ["…"]
  },
  "fact_check_report_id": "…",
  "overall_score": 0.92
}
```

---

## Agent → Table Mapping and Pointers

- ResearchCoordinatorAgent
  - Tables: `topic_ideas`, `research_sources`
  - Pointers: `topic_ids`, `source_ids`, `topics_count`, `sources_count`
  - UI behavior: show top N items with “Load more”; fetch via endpoints.

- ContentCuratorAgent
  - Tables: `topic_ideas` (marks `is_selected=true`), `pipeline_runs.chosen_topic_id`
  - Pointers: `chosen_topic_id`
  - UI behavior: show chosen topic details; fetch via topics endpoint.

- SEOStrategistAgent
  - Table: `content_outlines`
  - Pointers: `outline_id`, `sections_count`
  - UI behavior: show outline sections via outlines endpoint.

- ContentWriterAgent
  - Table: `content_drafts`
  - Pointers: `content_draft_id`, `version`, `stage`
  - UI behavior: link to Content tab for the draft; do not embed full body.

- QualityGateAgents (FactCheck, DomainExpertise, StyleCritic, Compliance)
  - Tables: `quality_assessments`; fact‑check details in `fact_check_reports`
  - Pointers: `quality_assessment_ids_by_gate`, `fact_check_report_id`, `overall_score`
  - UI behavior: per‑gate scores with expandable details; fetch via quality endpoint.

- ContentEditorAgent
  - Table: `content_drafts` (new version, stage=final)
  - Pointers: `content_draft_id`, `version`, `stage`
  - UI behavior: link to Content tab; avoid embedding content.

- CritiqueAgent
  - Table: `critique_reports`
  - Pointers: `critique_report_id`, `cycle`, `decision`
  - UI behavior: show decision summary; fetch full report via quality endpoint.

- ContentFormatterAgent
  - Table: `channel_content` (unpublished)
  - Pointers: `channel_content_ids_by_platform`
  - UI behavior: show per‑platform preview (title/excerpt); fetch via content channel endpoint.

- HumanReviewGateAgent
  - Tables: `human_reviews`, `content_drafts` (stage=human_review_pending)
  - Pointers: `human_review_id`, `content_draft_id`, `review_status`
  - UI behavior: show status and link to Content tab.

- PublisherAgent
  - Tables: `channel_content` (published), `publications`, updates `content_drafts.published_urls`
  - Pointers: `publication_ids`, `published_urls`
  - UI behavior: show published URLs; fetch publication details via publishing endpoint.

---

## Backend Endpoints (Read‑Only)

Existing
- `GET /api/v1/content/sources/{run_id}` → `research_sources`
- `GET /api/v1/content/drafts/{run_id}` and `GET /api/v1/content/{content_id}` → `content_drafts`
- `GET /api/v1/pipeline/runs/{run_id}/topics` → `topic_ideas`

New (to add)
- Quality
  - `GET /api/v1/quality/assessments/{run_id}` → returns all `quality_assessments` by gate, optionally embedding `fact_check_reports` summary
  - Optional: `GET /api/v1/quality/critique/{report_id}` → returns one `critique_reports` record

- Content (formatting)
  - `GET /api/v1/content/outlines/{run_id}` or `GET /api/v1/content/outline/{outline_id}` → `content_outlines`
  - `GET /api/v1/content/channels/{run_id}` or `GET /api/v1/content/channel/{id}` → `channel_content`

- Review (DB‑backed)
  - `GET /api/v1/review/by-run/{run_id}` → latest `human_reviews` for the run

- Publishing
  - `GET /api/v1/publishing/publications/{run_id}` → `publications` joined with `channel_content`

Notes
- No schema changes. All endpoints are read‑only views on existing tables.
- Where helpful, endpoints can include compact summaries for UI cards (e.g., first 3 outline sections).

---

## Frontend: Agent Log UI Changes

Detection
- When a log’s `context` includes known pointer keys, render controls to “View details” instead of dumping raw JSON.

Behavior by pointer type
- Topics/Sources: call the existing topics/sources endpoints and show first 3 items with “Load more.”
- Outline: call outlines endpoint and show section titles/bullets.
- Quality: call assessments endpoint; show per‑gate score chips; expand to lists of suggestions/evidence; include fact‑check claims summary when available.
- Channel Content / Publications: show per‑platform titles/excerpts and URLs.
- Writing/Editing: show a link to `/pipelines/{runId}?tab=content&draftId={content_draft_id}` (no excerpts in logs).
- Human Review: show status and link to Content tab with the current draft selected.

Constraints
- Keep log expansion light: paginate or limit default items to 3; use “Load more” to fetch additional items.
- Preserve current JSON “Context” panel as a fallback (toggle) for debugging.

---

## Logging Contract (Context Keys)

Standardize these keys in `log_agent_activity` calls:

- Research: `topics_count`, `sources_count`, `topic_ids[]`, `source_ids[]`
- Curation: `chosen_topic_id`
- SEO: `outline_id`, `sections_count`
- Writing/Editing: `content_draft_id`, `version`, `stage`
- Quality Gates: `quality_assessment_ids{gate:[]}`, `fact_check_report_id`, `overall_score`
- Critique: `critique_report_id`, `cycle`, `decision`
- Formatting: `channel_content_ids{platform:[]}`
- Human Review: `human_review_id`, `content_draft_id`, `review_status`
- Publishing: `publication_ids[]`, `published_urls[]`

These are additive to the existing human‑readable `message` and small metrics; no large blobs.

---

## UX Notes

- Full content bodies are not embedded in logs. For writing and editing, use a link to the Content tab view for the draft.
- For research “inline preview,” default to showing 3 topics (title, domain, overall_score, 3 keywords) and 3 sources (title, domain, url). Provide a “Load more” action.
- Keep the existing auto‑refresh and filters. Expanded entries fetch details independently and should handle transient errors gracefully.

---

## Rollout Plan

1) Backend
- Add the read‑only endpoints listed above.
- Update pipeline logging to include the standardized pointer keys.

2) Frontend
- Update Agent Log component to detect pointer keys and fetch details lazily.
- Add links to Content tab for draft‑related logs.

3) Validation
- Verify that logs remain small (pointer‑only).
- Expand each stage in Agent Log to confirm details render correctly and links navigate to the right draft.

---

## Open Questions / Defaults

- Research preview default: top 3 topics and sources; “Load more” fetches additional items.
- Quality detail: show per‑gate chips and allow expansion for suggestions/evidence; include fact‑check claims summary counts.

