# Overall LangGraph Agent Orchestration

The ContentRunway system uses **LangGraph StateGraph** to orchestrate a sophisticated multi-agent pipeline for automated content creation. The architecture follows a **ReAct (Reasoning + Acting)** pattern with conditional routing and quality gates.

---

## Key Orchestration Features

- **State-driven execution** with a comprehensive `ContentPipelineState` that tracks the entire pipeline.  
- **Conditional routing** based on quality scores, approval status, and error conditions.  
- **Parallel execution** of quality gates for efficiency.  
- **Retry logic** with improvement guidance when quality thresholds aren't met.  
- **Human-in-the-loop** approval gates before publishing.  
- **Multi-tenant isolation** with tenant-specific configurations and checkpointing.  
- **Persistent checkpointing** using PostgreSQL for state recovery.

---

## Pipeline Flow

```
Research → Curation → SEO Strategy → Writing → Quality Gates (Parallel) →
Editing → Formatting → Human Review → Publishing
```

---

# Individual Agent Goals

## 1. Research Coordinator Agent
**Goal:** Orchestrate domain-specific research across multiple specialized researchers.  
**Function:** Manages parallel research execution, consolidates sources, generates topic ideas.  
**Tools:** Coordinates IT/Insurance, AI, Agentic AI, and AI Software Engineering research agents.

---

## 2. Domain-Specific Research Agents

### IT Insurance Research Agent
- **Goal:** Specialized research for IT/Insurance domain  
- **Tools:** Insurance journal search, regulatory updates, insurtech databases, cybersecurity tools  
- **Knowledge Areas:** Regulatory compliance, digital transformation, insurtech innovations, cybersecurity insurance

### AI Research Agent
- **Goal:** Research AI/ML developments and trends  
- **Focus:** Technical AI content, software engineering with AI, LLM integrations

### Agentic AI Research Agent
- **Goal:** Research multi-agent systems, LangGraph, and agent orchestration  
- **Focus:** ReAct patterns, tool calling, state management, agent architectures

---

## 3. Content Curator Agent
**Goal:** Select optimal topics from research results using relevance, novelty, and SEO difficulty scores.  
**Output:** Chosen topics with audience targeting and goal definition.

---

## 4. SEO Strategist Agent
**Goal:** Optimize content for search engines and social media.  
**Function:** Keyword suggestions, heading optimization, internal linking, social snippets.

---

## 5. Content Writer Agent
**Goal:** Create high-quality, long-form content (1,200–1,800 words) with proper citations.  
**Output:** Platform-agnostic draft with citation markers and structured abstract.

---

## 6. Quality Gate Agents (Parallel Execution)

### Fact-Checking Gate
- **Goal:** Verify factual accuracy against sources and external verification.  
- **Function:** Cross-reference claims with sources, external fact-checking, citation coverage analysis.

### Domain Expertise Gate
- **Goal:** Validate technical accuracy for domain-specific content.  
- **Function:** Technical depth assessment, currency checks, domain-specific validation.

### Style Critic Gate
- **Goal:** Ensure brand voice consistency and writing quality.  
- **Function:** Rubric-based critique on clarity, depth, originality, structure, tone.

### Technical Review Gate
- **Goal:** Technical accuracy and depth validation.  
- **Function:** Code example verification, technical concept accuracy.

### Compliance Gate
- **Goal:** Copyright, plagiarism, and legal compliance checking.  
- **Function:** Plagiarism detection, copyright verification, PII scanning.

---

## 7. Content Editor Agent
**Goal:** Apply quality improvements and ensure house style.  
**Function:** Paragraph-level edits, style consistency, link formatting.

---

## 8. Content Formatter Agent
**Goal:** Create platform-specific content variants.  
**Output:** Medium, personal blog, LinkedIn, Twitter-optimized versions.

---

## 9. Human Review Gate
**Goal:** Wait for human approval and capture feedback.  
**Function:** Present content for a 15-minute review window, capture inline edits, record approval decisions.

---

## 10. Publishing Agent
**Goal:** Distribute approved content across multiple platforms.  
**Platforms:** Medium, personal blog, LinkedIn, Twitter.  
**Function:** Platform-specific API calls, URL tracking, publication status monitoring.

---

# LLM Usage and Context

The system supports multiple LLM providers with provider-specific usage patterns and role-based model selection.

## Primary LLM Providers

### 1. OpenAI (GPT-4, GPT-4-turbo, GPT-4o-mini)
**Usage Context:**
- Primary content generation (Writer Agent)  
- Complex reasoning tasks (Research Coordinator)  
- Style and quality critique  
- Technical content validation

**Cost Optimization:** Use GPT-4o-mini for simpler tasks; GPT-4 for complex content creation.

---

### 2. Google AI (Gemini Pro, Gemini Flash)
**Usage Context:**
- Alternative content generation  
- Research synthesis and analysis  
- Fact-checking and verification tasks  
- Multi-modal content analysis (future)

---

### 3. Anthropic (Claude Sonnet, Claude Haiku)
**Usage Context:**
- Long-form content analysis  
- Complex reasoning and critique  
- Safety and compliance checking  
- Technical documentation review

---

## LLM Usage Patterns

### Content Generation
- **Primary:** OpenAI GPT-4 for high-quality content creation  
- **Fallback:** Google Gemini Pro or Anthropic Claude Sonnet  
- **Optimization:** Model selection based on content type and quality requirements

### Research and Analysis
- **Multi-provider approach:** Parallel research using different models for diverse perspectives  
- **Source validation:** Cross-verification using multiple LLMs  
- **Trend analysis:** Specialized prompting for domain-specific insights

### Quality Assurance
- **Fact-checking:** Multiple LLM cross-validation  
- **Style critique:** Model fine-tuned prompts for brand voice consistency  
- **Technical review:** Domain-specific model selection based on expertise

### Cost Management
- **Tiered approach:** Expensive models for critical tasks, cheaper models for preliminary work  
- **Token optimization:** Structured prompts to minimize token usage  
- **Caching:** Redis-based caching for repeated operations

---

## Multi-Tenant LLM Configuration
- **Per-tenant API keys:** Isolated billing and usage tracking  
- **Custom model preferences:** Tenant-specific LLM provider priorities  
- **Usage monitoring:** Real-time cost and token tracking per tenant  
- **Rate limiting:** Tenant-based throttling and budget controls

---

# Operational Principles

- The system **prioritizes quality over speed**, using the most appropriate LLM for each task while maintaining cost efficiency through intelligent model selection and caching strategies.  
- **Retry logic** includes guided improvement suggestions when thresholds are not met and records retries in the persistent `ContentPipelineState`.  
- **Tenant isolation** ensures configuration safety and independent checkpointing/restore for each tenant.

---

> **Note:** RetryClaude can make mistakes. Please double-check responses.
