# LangGraph Content Pipeline — Solution Strategy

## High level requirement

I want to create a system of ReAct-style agents using the LangGraph framework that helps me on a regular basis (daily) with:

1. Research and content idea generation
2. Content creation
3. Critique
4. Content finalization
5. Human review
6. Posting/publishing to social media (long posts on Medium and my blog; short posts on LinkedIn, X/Twitter)

---

## Solution strategy — overview

Design a LangGraph-powered, ReAct-style multi-agent system that runs daily to:

1. Research & ideate
2. Draft
3. Critique
4. Finalize
5. Request approval
6. Publish across Medium, your blog, LinkedIn, and X (Twitter)

Stack: **FastAPI** + **LangGraph** + **Postgres** + **Milvus**, Dockerized end-to-end.

---

## Goals (what “good” looks like)

- **Daily cadence:** scheduled pipeline generates 3–5 topic ideas, selects 1–2, produces platform-specific drafts, and stops for human approval before publishing.
- **Grounded & citeable:** research is source-linked; drafts preserve citations; fact-check & plagiarism checks run before approval.
- **Reusable voice:** consistent style tailored to your brand (e.g., "Budding Digital"), with tone variants per channel.
- **Traceable:** every step logged; artifacts (notes, outlines, drafts, rubrics, approvals, posts) stored and searchable.

---

## High-level architecture

### Frontend (Next.js)
- Dashboard: queue of daily runs, diffs between draft → final, one-click approve/publish, edit-in-place.
- Library: ideas, sources, outlines, drafts, published URLs.

### Backend (FastAPI)
- `/runs/*` APIs (create, status, artifacts), `/publish/*`, `/webhooks/*` for Medium/WordPress callbacks.
- Auth (JWT), RBAC (you + collaborators).

### Orchestration (LangGraph workers)
- A graph per “content run” with tool-calling ReAct agents.
- Redis (optional) for queues / rate-limit cache.

### Data
- **Postgres** for metadata, runs, approvals, schedules.
- **Milvus** for long-term knowledge: past posts, notes, PDFs, transcripts, web snapshots → retrieval for context.
- Object store (S3 or local volumes) for artifacts (PDFs, HTML snapshots).

### Scheduling
- APScheduler / Celery beat / cron to kick off runs (example: 08:30 ET daily).

### Observability
- Structured logging (JSON), LangSmith / OpenAI eval traces (optional), Prometheus + Grafana (optional).

---

## Agent roster (ReAct roles)

1. **Researcher**
   - Tools: web search, web scrapers, RSS / YouTube transcripts, arXiv/API, PDF parser.
   - Output: candidate topics (title + 2-sentence rationale), annotated source set (URL, quote, claim).

2. **Curator / Planner**
   - Chooses 1–2 topics using scores (relevance, novelty, SEO difficulty).
   - Produces outline + key claims + intended audience & goal.

3. **SEO Strategist**
   - Suggests keywords, headings, FAQ, internal links (to your past posts) and social snippets; updates outline.

4. **Writer**
   - Drafts platform-agnostic long-form (1,200–1,800 words) and embeds citation markers ([1], [2]).
   - Emits structured abstract used for social short-form.

5. **Fact-Checker**
   - Cross-checks claim→source alignment; flags weak/uncited claims.
   - Produces a "citation fix list" for the Writer or softens claims.

6. **Style Critic**
   - Rubric-based critique (clarity, depth, originality, structure, tone).
   - Suggests concrete edits; can reject if below threshold.

7. **Editor / Finalizer**
   - Applies fixes at paragraph level; ensures house style, link formatting, image suggestions/captions.

8. **Compliance / Plagiarism Gate**
   - Runs plagiarism checks (optional API) + copyright pass; verifies image rights.

9. **Channel Formatter & Publisher**
   - Produces per-channel variants:
     - **Medium:** canonical URL, tags, TL;DR, lead image.
     - **Your blog (WordPress / Next):** front-matter / metadata, SEO fields.
     - **LinkedIn:** 1,200–2,200 chars, hashtags, hook, CTA.
     - **X/Twitter:** 1–3 posts (threads) with links + alt text.
   - Calls publishing APIs (dry-run until approved).

10. **Human Review Gate**
    - Waits for your approval in the dashboard; records inline edits as training signals.

---

## Graph design (LangGraph)

### State schema (Pydantic)

```python
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional

class Source(BaseModel):
    url: HttpUrl
    title: str
    snippet: str
    claim_ids: List[str]

class TopicIdea(BaseModel):
    id: str
    title: str
    rationale: str
    score: float

class Outline(BaseModel):
    title: str
    audience: str
    goal: str
    headings: List[str]
    key_claims: Dict[str, str]  # id -> claim text

class Draft(BaseModel):
    markdown: str
    citations: Dict[str, List[HttpUrl]]  # claim_id -> sources
    abstract: str

class ChannelDrafts(BaseModel):
    medium_md: str
    blog_md: str
    linkedin_text: str
    twitter_posts: List[str]
    images: List[str]  # asset keys

class RunState(BaseModel):
    run_id: str
    topics: List[TopicIdea] = []
    chosen_topic_id: Optional[str] = None
    sources: List[Source] = []
    outline: Optional[Outline] = None
    draft: Optional[Draft] = None
    channel_drafts: Optional[ChannelDrafts] = None
    critique_notes: List[str] = []
    fact_check_report: Dict[str, str] = {}
    compliance_report: Dict[str, str] = {}
    approval_required: bool = True
    approved: bool = False
    publish_urls: Dict[str, str] = {}
```

### Nodes
`researcher_node`, `curator_node`, `seo_node`, `writer_node`, `factcheck_node`, `critic_node`, `editor_node`, `compliance_node`, `formatter_node`, `approval_node`, `publisher_node`.

### Conditional edges
- If `fact_check_report` has blocking items → back to `writer_node`.
- If `critic_node.score` < threshold → back to `writer_node` or `editor_node`.
- If `approved == False` → wait on `approval_node`; resume to `publisher_node` once approved.

### Persistence
Persist `RunState` after each node to Postgres (JSONB) and store large artifacts in S3 / object store.

---

## Tooling & integrations

### Research tools
- Search: SerpAPI / Tavily (or other search APIs).
- Scrape: `trafilatura`, `readability-lxml`.
- PDF parsing: `pypdf`, `pdfplumber`.
- YouTube transcripts: `youtube-transcript-api`.
- Feeds: RSS polling.
- Deduping: URL canonicalization + cosine similarity on embeddings.

### Knowledge base (Milvus)
- Collections: `posts`, `notes`, `clips`, `citations`.
- Use retrieval-augmented prompts (RAG) to ground drafts in your previous work & saved highlights.

### Publishing
- **Medium API** (token & publicationId required).
- **Blog**: WordPress REST (JWT) or your Next blog API.
- **LinkedIn**: v2 UGC Posts (requires permissions).
- **X/Twitter**: v2 API (paid) or a scheduling service (Buffer / n8n / Zapier) as fallback.

### Safety
- Plagiarism: third-party API or cosine-similarity checks vs KB.
- Copyright: enforce max-quote-length & image usage checks.
- PII: automatic scan for sensitive info before posting.

---

## End-to-end daily flow (happy path)

1. **Kickoff** (08:30 ET) → new `run_id`.
2. **Researcher:** pulls feeds + searches, extracts 3–5 topic ideas, saves sources.
3. **Curator/Planner:** scores & selects 1–2 ideas, drafts outline (audience, goal, headings, claim IDs).
4. **SEO Strategist:** injects keywords, schema/FAQ suggestions, internal links to prior posts.
5. **Writer:** creates long-form draft (Markdown) with citation markers `[C-123]` mapped to sources.
6. **Fact-Checker:** verifies each claim; produces fix list or passes.
7. **Style Critic:** rubric scores (≥85 required), returns actionable edits if not met.
8. **Editor/Finalizer:** applies fixes, ensures consistent voice, suggests images.
9. **Formatter:** produces Medium / Blog / LinkedIn / X variants.
10. **Human Gate:** you review in dashboard; inline edits captured; click **Approve**.
11. **Publisher:** publishes & records URLs; posts social teasers with canonical links.
12. **Post-run analytics:** store metrics, ask user “was this helpful?” for reinforcement.

---

## Data model (Postgres)
- `runs(id, created_at, status, approved_at, owner_id)`
- `artifacts(id, run_id, kind, path, sha256, created_at)`
- `sources(id, run_id, url, title, snippet, claim_ids[])`
- `ideas(id, run_id, title, rationale, score)`
- `outlines(id, run_id, jsonb)`
- `drafts(id, run_id, markdown, abstract, citations_jsonb)`
- `channel_drafts(id, run_id, medium_md, blog_md, linkedin_text, twitter_jsonb)`
- `reviews(id, run_id, reviewer, notes_jsonb, rubric_jsonb, decision)`
- `publishes(id, run_id, channel, payload_jsonb, url, status, error)`

---

## Prompt skeletons (short & specific)

### Researcher (system)
> You are a research analyst. Produce 3–5 timely topic ideas for {audience} in {domain}. Each idea must cite at least 2 credible sources with short quotes mapping to claim IDs. Return JSON.

### Writer (system)
> You are a senior writer. Using `outline`, `key_claims`, and `sources`, draft a 1,200–1,800 word article in my voice (see `STYLE_GUIDE`). Keep statements grounded; insert citation markers `[C-<id>]`. Return Markdown and an abstract.

### Fact-Checker (system)
> Verify each claim id → source URLs. Flag unsupported or weak claims. Either propose a fix with updated text + new citation or mark “OK”.

### Style Critic (system)
> Score on clarity, depth, originality, structure, tone (0–100). Provide actionable edit list. If score < 85, block progression.

### Formatter (system)
> Produce per-channel outputs: Medium MD (with tags), Blog MD (front-matter), LinkedIn post (≤ 2,200 chars), X thread (1–3 tweets). Include alt text for images.

(Keep a local `STYLE_GUIDE` document with examples of your voice and tone.)

---

## LangGraph code skeleton

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(RunState)

graph.add_node("research", researcher_node)
graph.add_node("curate", curator_node)
graph.add_node("seo", seo_node)
graph.add_node("write", writer_node)
graph.add_node("factcheck", factcheck_node)
graph.add_node("critic", critic_node)
graph.add_node("edit", editor_node)
graph.add_node("compliance", compliance_node)
graph.add_node("format", formatter_node)
graph.add_node("approval", approval_node)
graph.add_node("publish", publisher_node)

graph.set_entry_point("research")
graph.add_edge("research", "curate")
graph.add_edge("curate", "seo")
graph.add_edge("seo", "write")
graph.add_edge("write", "factcheck")

def needs_rewrite(state):
    return "write" if state.fact_check_report.get("blocking") else "critic"

graph.add_conditional_edges("factcheck", needs_rewrite, {"write": "write", "critic": "critic"})

def needs_edits(state):
    return "edit" if state.critique_notes and state.critique_notes[0].startswith("BLOCK") else "compliance"

graph.add_conditional_edges("critic", needs_edits, {"edit": "edit", "compliance": "compliance"})

graph.add_edge("edit", "compliance")
graph.add_edge("compliance", "format")
graph.add_edge("format", "approval")

def go_publish(state):
    return "publish" if state.approved else "approval"

graph.add_conditional_edges("approval", go_publish, {"publish": "publish", "approval": "approval"})

graph.add_edge("publish", END)

app = graph.compile(checkpointer=MemorySaver())
```

(Each node calls an Agent executor + toolset, updates `RunState`, persists to DB.)

---

## Deployment (Docker Compose)

```yaml
version: "3.9"
services:
  api:
    build: ./backend
    env_file: .env
    ports: ["8000:8000"]
    depends_on: [db, milvus]
  worker:
    build: ./worker  # LangGraph runner
    env_file: .env
    depends_on: [api, db, milvus, redis]
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: contentops
      POSTGRES_USER: contentops
      POSTGRES_PASSWORD: contentops
    volumes: ["pgdata:/var/lib/postgresql/data"]
    ports: ["5432:5432"]
  milvus:
    image: milvusdb/milvus:2.4.7
    ports: ["19530:19530", "9091:9091"]
  redis:
    image: redis:7
    ports: ["6379:6379"]
volumes:
  pgdata:
```

---

## Scheduling & approvals
- APScheduler job `daily_run` creates a `RunState` and enqueues graph execution.
- Worker pauses at `approval_node` (persist state = `awaiting_approval`).
- Frontend “Approve & Publish” calls `/runs/{id}/approve` → flips `approved=True`; worker resumes.

---

## Observability & quality
- Rubrics: clarity, depth, originality, correctness, usefulness; tracked over time.
- A/B: test hooks/introductions; track CTR from LinkedIn/X to canonical article.
- Metrics: time per node, rewrite loops, req/resp tokens, publish outcomes.
- Safety: max quote length, automated link check (HTTP 200), alt-text completeness.

---

## Risks & mitigations
- LinkedIn/X API gating → fallback to Buffer / n8n for posting.
- Hallucinations → strict fact-check block and citation-first prompting.
- Rate limits → cache search results; exponential backoff.
- Voice drift → keep `STYLE_GUIDE` and few-shot exemplars from your best posts.

---

## MVP milestones (suggested)
1. **Week 1:** Research → Outline → Draft pipeline with Milvus grounding; manual publish.
2. **Week 2:** Fact-check, Critic, Editor loops + approval UI.
3. **Week 3:** Channel formatting + Medium/Blog publish; LinkedIn/X via Buffer.
4. **Week 4:** Analytics, rubrics, RSS ingestion, image pipeline (captions/alt text).

---

## Example API contracts (FastAPI)

```python
@app.post("/runs")
def start_run(payload: StartRun): ...

@app.get("/runs/{run_id}")
def get_run(run_id: str): ...

@app.post("/runs/{run_id}/approve")
def approve(run_id: str): ...

@app.post("/publish/{run_id}")
def publish(run_id: str, channels: List[str]): ...
```

---

## Environment (.env) checklist
- `OPENAI_API_KEY` (and/or `GOOGLE_API_KEY` for Gemini)
- `TAVILY_API_KEY` / `SERPAPI_KEY`
- `MEDIUM_TOKEN`, `MEDIUM_PUBLICATION_ID`
- `WP_BASE_URL`, `WP_USER`, `WP_APP_PASSWORD` (if WordPress)
- `LINKEDIN_CLIENT_ID/SECRET`, `LINKEDIN_ACCESS_TOKEN` (or Buffer creds)
- `TWITTER_BEARER_TOKEN` (or Buffer creds)
- `DATABASE_URL`, `MILVUS_HOST/PORT`, `REDIS_URL`

---

*Document generated from the user-provided requirements and solution approach.*

