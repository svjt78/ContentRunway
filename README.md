# ContentRunway - AI Content Pipeline

ContentRunway is a quality-first AI content creation system that uses LangGraph to orchestrate ReAct-style agents for producing high-quality, domain-specific content in IT/Insurance/AI domains.

## Features

### 🎯 Quality-First Approach
- Multi-layered quality validation with 85%+ thresholds
- Parallel quality gates (fact-check, domain expertise, style, compliance)
- Human review workflow with 15-minute approval process

### 🔬 AI-Powered Research
- Domain-specific research agents
- Multi-source content aggregation
- Citation management and fact-checking
- Vector similarity search with Milvus

### 🛠️ LangGraph Orchestration
- ReAct agent patterns with OpenAI integration
- State-driven pipeline execution
- Parallel processing for efficiency
- Checkpointing and recovery

### 📊 Comprehensive Dashboard
- Real-time pipeline monitoring
- Quality score tracking
- Performance analytics
- Content review interface

## Technology Stack

### Backend
- **FastAPI** - Python REST API
- **LangGraph** - Agent orchestration with StateGraph
- **PostgreSQL** - Structured data with RLS
- **Milvus** - Vector database for knowledge base
- **Redis** - Caching and session management

### Frontend
- **Next.js 14** - App Router with TypeScript
- **Tailwind CSS** - Styling and design system
- **Zustand** - Global state management
- **TanStack Query** - Server state management
- **Socket.io** - Real-time updates

### Infrastructure
- **Docker** - Containerization with hot reloading
- **Docker Compose** - Development environment
- **Nginx** - Reverse proxy

## Getting Started

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- (Optional) Google AI and Anthropic API keys

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ContentRunway
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start the development environment**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Environment Variables

Required environment variables in `.env`:

```bash
# OpenAI Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional AI Providers
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Publishing Platforms
MEDIUM_API_KEY=your_medium_api_key_here
MEDIUM_USER_ID=your_medium_user_id_here

# Security
JWT_SECRET=your_jwt_secret_here
```

## Architecture Overview

### Pipeline Flow

1. **Research Phase** - Domain-specific agents gather sources in parallel
2. **Planning Phase** - Content curation → SEO strategy → outline creation
3. **Writing Phase** - AI content generation with citations
4. **Quality Gates** - Parallel validation (fact-check, domain expertise, style, compliance)
5. **Human Review** - 15-minute approval interface with inline editing
6. **Publishing Phase** - Multi-platform content distribution

### Quality Gates (85%+ threshold)

- **Factual Accuracy Gate** - Validates claims against sources (95% threshold)
- **Domain Expertise Gate** - Technical accuracy for IT/Insurance/AI (90% threshold)
- **Style Consistency Gate** - Brand voice alignment (88% threshold)
- **Technical Depth Gate** - Content expertise level (85% threshold)
- **Compliance Gate** - Regulatory and copyright compliance (95% threshold)

### Supported Domains

- **IT Insurance** - Cybersecurity, insurtech, digital transformation
- **AI Research** - Technical AI content, ML developments, LLM integrations
- **Agentic AI** - Multi-agent systems, LangGraph, agent orchestration
- **AI Software Engineering** - AI in development, code generation

### Publishing Platforms

- **Primary**: Medium, Personal Blog
- **Future**: LinkedIn, Twitter

## Development

### Project Structure

```
ContentRunway/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic
│   │   ├── core/           # Configuration
│   │   └── db/             # Database setup
│   ├── migrations/         # Database migrations
│   └── requirements.txt
├── frontend/               # Next.js application
│   ├── src/
│   │   ├── app/            # App Router pages
│   │   ├── components/     # React components
│   │   ├── lib/            # Utilities and API
│   │   └── hooks/          # Custom hooks
│   └── package.json
├── langgraph/              # LangGraph agents and workflows
│   └── contentrunway/
│       ├── agents/         # Agent implementations
│       ├── tools/          # Agent tools
│       ├── state/          # Pipeline state definitions
│       └── pipeline.py     # Main pipeline orchestration
├── docker-compose.yml      # Development environment
└── docs/                   # Documentation
```

### Development Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart a service
docker-compose restart backend

# Run database migrations
docker-compose exec backend alembic upgrade head

# Access database
docker-compose exec postgres psql -U contentrunway -d contentrunway

# Frontend development (outside Docker)
cd frontend && npm install && npm run dev
```

### Hot Reloading

All services support hot reloading in development:
- Frontend: Next.js development server with file watching
- Backend: Uvicorn with `--reload` for FastAPI changes
- Database: Local volume mounting for persistence

## Phase 1 Implementation

This is Phase 1 implementation focused on:
- ✅ Single-tenant personal use
- ✅ Quality-first content pipeline
- ✅ Docker containerization with hot reloading
- ✅ Core agent orchestration with LangGraph
- ✅ PostgreSQL and Milvus integration
- ✅ Next.js dashboard interface

**Phase 2 (Future)** will include:
- Multi-tenant architecture
- Advanced publishing workflows
- Performance analytics
- Mobile app
- Advanced AI model integrations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Create an issue in the repository
- Check the documentation in the `/docs` folder
- Review the API documentation at `/docs` when running locally