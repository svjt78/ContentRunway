# ContentRunway - AI Content Pipeline

## Project Overview
ContentRunway is a quality-first AI content creation system that uses LangGraph to orchestrate ReAct-style agents for producing high-quality, domain-specific content in IT/Insurance/AI domains.

**Phase 1**: Single-tenant personal use (no multi-tenant features)
**Focus**: Quality over speed with 85%+ quality thresholds

## Technology Stack

### Backend
- **FastAPI** - Python REST API
- **LangGraph** - Agent orchestration with StateGraph
- **PostgreSQL** - Structured data with RLS
- **Milvus** - Vector database for knowledge base
- **Redis** - Caching and session management

### Frontend
- **Next.js 14** - App Router with TypeScript
- **Tailwind CSS** - Styling
- **Zustand** - Global state management
- **TanStack Query** - Server state management
- **Monaco Editor** - Content editing
- **Socket.io** - Real-time updates

### Infrastructure
- **Docker** - Containerization with hot reloading
- **Docker Compose** - Development environment
- **PostgreSQL 16** - Database with local volumes
- **Nginx** - Reverse proxy

## Development Commands

### Setup
```bash
# Start development environment
docker-compose up -d

# Hot reload frontend
npm run dev

# Run backend with hot reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Frontend tests
npm test

# Backend tests
pytest

# Type checking
npm run type-check
mypy .
```

### Quality Checks
```bash
# Lint frontend
npm run lint

# Lint backend
ruff check .
black .

# Format code
prettier --write .
```

## Key Features

### AI Pipeline Flow
1. **Research Phase** - Domain-specific agents gather sources
2. **Planning Phase** - Content curation → SEO strategy → outline
3. **Writing Phase** - AI content generation with citations
4. **Quality Gates** - Parallel validation (fact-check, domain expertise, style, compliance)
5. **Human Review** - 15-minute approval interface
6. **Publishing Phase** - Multi-platform distribution

### Quality Gates (85%+ threshold)
- Technical accuracy validation (90%+)
- Domain expertise verification (90%+)
- Style consistency checking (88%+)
- Compliance validation (95%+)
- Overall quality minimum (85%+)

## Project Structure

```
ContentRunway/
├── backend/                 # FastAPI application
├── frontend/               # Next.js application
├── langgraph/              # LangGraph agents and workflows
├── docker-compose.yml      # Development environment
├── Dockerfile.backend      # Backend container with hot reload
├── Dockerfile.frontend     # Frontend container with hot reload
└── docs/                   # Documentation
```

## Docker Development

All services run in Docker with:
- Hot reloading enabled for development
- Database volumes mounted locally for persistence
- Environment variables for API keys and configuration
- Health checks and service dependencies

## LLM Integration

### Supported Providers
- **OpenAI** (GPT-4, GPT-4-turbo, GPT-4o-mini) - Primary content generation
- **Google AI** (Gemini Pro, Gemini Flash) - Research and analysis
- **Anthropic** (Claude Sonnet, Claude Haiku) - Long-form analysis and safety

### Usage Patterns
- Quality-first model selection
- Cost optimization through intelligent model routing
- Multi-provider research for diverse perspectives
- Redis caching for repeated operations

## Content Domains

### Specialized Research Agents
- **IT Insurance** - Regulatory compliance, digital transformation, insurtech
- **AI Research** - Technical AI content, ML developments, LLM integrations  
- **Agentic AI** - Multi-agent systems, LangGraph, agent orchestration

### Publishing Platforms
- **Primary**: Personal Blog

## Development Notes

- Build dockerized solution with hot reloading
- Mount database volumes locally for persistence
- Single-tenant implementation (Phase 1)
- Quality thresholds enforced at every gate
- 15-minute human review workflow
- Comprehensive logging and monitoring
- make sure to use uv to manage all dependencies