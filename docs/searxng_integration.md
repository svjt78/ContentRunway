# SearXNG Integration for Trending Content Discovery

## Overview

ContentRunway now integrates SearXNG to discover trending content ideas from social platforms (Twitter/LinkedIn) using creative search strings. This complements the existing RSS feeds and arXiv academic sources.

## Architecture

### Components Added

1. **SearXNG Docker Service** - Self-hosted search engine
2. **SearXNGTool** - Python tool for programmatic search  
3. **Enhanced WebResearchTool** - Combines traditional + trending sources
4. **Updated ResearchCoordinatorAgent** - Orchestrates mixed research

### Search Strategy

The integration uses creative search strings to find trending content:

```
# Example Twitter search for Agentic AI domain:
site:twitter.com "multi-agent systems" trending discussions
site:twitter.com "agentic AI" popular posts latest  
site:twitter.com "AI agents" viral discussions trending

# Example LinkedIn search for AI Software Engineering:
site:linkedin.com "AI development tools" trending discussions
site:linkedin.com "automated coding" popular insights
site:linkedin.com "AI-assisted development" industry trends
```

## Configuration

### Environment Variables

```bash
# SearXNG Configuration
SEARXNG_BASE_URL=http://localhost/search
SEARXNG_SECRET_KEY=your-searxng-secret-key-here
SEARXNG_ENABLED=true
SEARXNG_RATE_LIMIT=60  # searches per hour
```

### Docker Services

SearXNG runs as a Docker service proxied through nginx:
- SearXNG: `http://localhost:8080` (internal)
- Nginx proxy: `http://localhost/search/` (external)

## Usage

### Programmatic Search

```python
from langgraph.contentrunway.tools.searxng_tool import SearXNGTool

# Initialize tool
searxng = SearXNGTool("http://localhost/search")

# Search trending content
trending_content = await searxng.search_trending_content(
    domain="agentic_ai",
    platforms=["twitter", "linkedin"],
    max_results_per_platform=10
)

# Generate creative search queries
queries = await searxng.generate_creative_search_queries(
    base_topic="AI automation",
    domain="ai_software_engineering"
)
```

### Enhanced Research

```python
from langgraph.contentrunway.tools.web_research import WebResearchTool

# Initialize enhanced web research
web_research = WebResearchTool("http://localhost/search")

# Combined traditional + trending search
results = await web_research.enhanced_domain_search(
    query="multi-agent systems",
    domain="agentic_ai", 
    include_trending=True,
    max_sources=20
)
```

### Research Pipeline Integration

The ResearchCoordinatorAgent automatically includes trending content:

```python
# Trending content is included by default
research_results = await research_coordinator.execute(
    query="AI automation trends",
    domains=["agentic_ai"],
    state=pipeline_state
)

# Disable trending content if needed
state.config_overrides['include_trending'] = False
```

## Content Scoring

### Traditional Sources
- Relevance: 40%
- Credibility: 40% 
- Engagement: 10%
- Recency: 10%

### Trending Sources  
- Relevance: 30%
- Credibility: 20%
- Engagement: 30%
- Recency: 20%

### Credibility Scores
- Academic papers (arXiv): 0.95
- Industry publications: 0.80-0.85
- LinkedIn content: 0.70
- Twitter content: 0.60

## Testing

### Start Services
```bash
docker-compose up -d
```

### Run Integration Tests
```bash
python test_searxng_integration.py
```

### Run Code Validation
```bash  
python validate_searxng_code.py
```

## Search Templates

### IT Insurance
- Cyber insurance trends and discussions
- Insurtech innovation and digital transformation
- IT security insurance insights

### AI
- Artificial intelligence industry trends  
- Machine learning business applications
- AI ROI and transformation discussions

### Agentic AI
- Multi-agent systems and autonomous agents
- Agent orchestration and LangGraph discussions
- Agent-based system innovations

### AI Software Engineering
- AI development tools and coding assistance
- Automated development and productivity
- AI-assisted development trends

## Rate Limiting

- Default: 60 searches per hour
- Configurable via `SEARXNG_RATE_LIMIT`
- Built-in backoff and retry logic
- Graceful degradation when limits exceeded

## Troubleshooting

### Common Issues

1. **SearXNG not responding**: Check `docker-compose ps` and ensure searxng service is running
2. **Nginx proxy errors**: Verify nginx.conf includes `/search/` location block
3. **No trending results**: Check SEARXNG_ENABLED=true in environment
4. **Rate limit errors**: Increase SEARXNG_RATE_LIMIT or implement caching

### Debug Mode

Enable debug logging to troubleshoot search issues:

```python
import logging
logging.getLogger('langgraph.contentrunway.tools.searxng_tool').setLevel(logging.DEBUG)
```

## Future Enhancements

- [ ] Add Reddit search templates
- [ ] Implement search result caching
- [ ] Add trending topic trend analysis
- [ ] Integrate sentiment analysis for social content
- [ ] Add platform-specific engagement metrics