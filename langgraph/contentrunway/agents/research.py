"""Research Coordinator Agent - Orchestrates domain-specific research."""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import asyncio
import logging
from datetime import datetime
import uuid
import json

from ..state.pipeline_state import Source, TopicIdea, ContentPipelineState
from ..tools.web_research import WebResearchTool

logger = logging.getLogger(__name__)


# Research Coordinator Agent Instructions
research_agent_role_and_goal = """
You are a Research Coordinator Agent specializing in orchestrating comprehensive research across multiple domains (IT Insurance, AI, Agentic AI, AI Software Engineering).
Your primary goal is to coordinate parallel research activities, generate domain-specific queries, collect high-quality sources, and synthesize research findings into ranked topic ideas for content creation.
"""

research_agent_hints = """
Research best practices:
- Generate domain-specific queries that capture recent trends and practical applications
- Prioritize sources by credibility score and relevance to target domains
- Balance traditional academic sources with trending industry insights
- Focus on actionable, professional-level content opportunities
- Ensure comprehensive coverage across all specified domains
- Validate source credibility and enhance top sources with full content
- Generate topic ideas that support 1200-1800 word in-depth articles
- Score topics based on relevance, novelty, SEO difficulty, and content depth potential
"""

research_agent_output_description = """
The Research Coordinator Agent returns a comprehensive research package containing:
- sources: List of up to 50 prioritized, credible sources with enhanced content
- topics: List of up to 10 ranked topic ideas with detailed scoring
- domain_queries: Map of domain-specific research queries generated
- research_summary: Metadata about research scope and results

Each source includes credibility_score, relevance_score, domain, title, summary, and optionally full content.
Each topic includes relevance_score, novelty_score, seo_difficulty, overall_score, and comprehensive metadata.
"""

research_agent_chain_of_thought_directions = """
Research coordination workflow:
1. Use '_generate_domain_queries' to create targeted research queries for each specified domain
2. Execute parallel research using '_execute_domain_research' for each domain with domain-specific agents
3. Use 'WebResearchTool.enhanced_domain_search' to gather both traditional and trending sources
4. Apply 'WebResearchTool.fetch_article_content' to enhance top sources with full content
5. Use 'WebResearchTool.validate_source_credibility' to score source reliability
6. Consolidate and deduplicate sources from all domains using URL and title matching
7. Use '_generate_topic_ideas' to synthesize research into compelling topic ideas
8. Apply '_score_topics' to rank topics using multi-criteria scoring (relevance, novelty, SEO, depth)
9. Return structured research package with sources limited to top 50 and topics limited to top 10

Tool usage conditions:
- Use WebResearchTool.enhanced_domain_search when include_trending=True in state config
- Use domain-specific agents (ITInsuranceResearchAgent, AIResearchAgent, etc.) when available
- Apply parallel execution with asyncio.gather for performance optimization
- Implement fallback queries if AI-generated domain queries fail JSON parsing
"""

research_agent_instruction = f"""
{research_agent_role_and_goal}
{research_agent_hints}
{research_agent_output_description}
{research_agent_chain_of_thought_directions}
"""


class ResearchCoordinatorAgent:
    """
    Coordinates research across multiple domain-specific research agents.
    Uses OpenAI GPT-4 for intelligent research orchestration and topic generation.
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,  # Balanced creativity for research
            max_tokens=4000
        )
        self.web_research_tool = WebResearchTool()
        
        # Domain-specific research agents
        self.domain_agents = {
            "it_insurance": ITInsuranceResearchAgent(),
            "ai": AIResearchAgent(), 
            "agentic_ai": AgenticAIResearchAgent(),
            "ai_software_engineering": AISoftwareEngineeringAgent()
        }
    
    async def execute(
        self,
        query: str,
        domains: List[str],
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """
        Execute coordinated research across specified domains.
        
        Args:
            query: Research query or topic area
            domains: List of domains to research
            state: Current pipeline state
            
        Returns:
            Dictionary containing sources and generated topic ideas
        """
        logger.info(f"Starting research coordination for domains: {domains}")
        
        try:
            # Step 0: Check for existing research in Milvus
            existing_sources = await self._check_existing_research(query, domains)
            
            # Step 1: Generate research queries for each domain
            domain_queries = await self._generate_domain_queries(query, domains)
            
            # Step 2: Execute parallel research across domains
            research_tasks = []
            include_trending = state.get('config_overrides', {}).get('include_trending', True)
            
            for domain, domain_query in domain_queries.items():
                if domain in self.domain_agents:
                    task = self._execute_domain_research(domain, domain_query, include_trending)
                    research_tasks.append(task)
            
            # Wait for all research to complete
            research_results = await asyncio.gather(*research_tasks, return_exceptions=True)
            
            # Step 3: Consolidate sources and store in Milvus
            all_sources = list(existing_sources)  # Start with existing sources
            for result in research_results:
                if isinstance(result, dict) and 'sources' in result:
                    all_sources.extend(result['sources'])
            
            # Remove duplicates by URL
            seen_urls = set()
            unique_sources = []
            for source in all_sources:
                url = source.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(source)
                elif not url and source.get('title'):  # Handle sources without URLs
                    unique_sources.append(source)
            
            # Store new sources in Milvus for future reference
            new_sources = [s for s in unique_sources if s not in existing_sources]
            if new_sources:
                await self._store_sources_in_milvus(new_sources, state["run_id"])
            
            all_sources = unique_sources
            
            # Step 4: Generate topic ideas from research
            topic_ideas = await self._generate_topic_ideas(all_sources, domains, query)
            
            # Step 5: Score and rank topics
            scored_topics = await self._score_topics(topic_ideas, all_sources)
            
            return {
                'sources': all_sources[:50],  # Limit to top 50 sources
                'topics': scored_topics[:10],  # Limit to top 10 topics
                'domain_queries': domain_queries,
                'research_summary': {
                    'total_sources': len(all_sources),
                    'domains_researched': len(domain_queries),
                    'topics_generated': len(topic_ideas)
                }
            }
            
        except Exception as e:
            logger.error(f"Research coordination failed: {e}")
            raise
    
    async def _generate_domain_queries(
        self,
        base_query: str,
        domains: List[str]
    ) -> Dict[str, str]:
        """Generate domain-specific research queries."""
        
        domain_descriptions = {
            "it_insurance": "IT and cyber insurance, insurtech, digital transformation in insurance",
            "ai": "Artificial intelligence, machine learning, AI applications and trends",
            "agentic_ai": "Multi-agent systems, LangGraph, ReAct patterns, agent orchestration",
            "ai_software_engineering": "AI in software development, code generation, AI-assisted development"
        }
        
        system_prompt = research_agent_instruction
        
        domain_list = "\n".join([f"- {domain}: {domain_descriptions.get(domain, '')}" for domain in domains])
        
        human_prompt = f"""Base research topic: "{base_query}"

Target domains:
{domain_list}

Generate a specific research query for each domain that will help find:
1. Recent developments and trends
2. Practical applications and use cases
3. Industry insights and expert opinions
4. Technical innovations and best practices

Format your response as JSON with domain names as keys and research queries as values.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            import json
            domain_queries = json.loads(response.content)
            
            # Fallback to basic queries if parsing fails
            if not isinstance(domain_queries, dict):
                domain_queries = {domain: f"{base_query} in {domain}" for domain in domains}
                
        except json.JSONDecodeError:
            # Fallback to basic queries
            domain_queries = {domain: f"{base_query} in {domain}" for domain in domains}
        
        return domain_queries
    
    async def _execute_domain_research(
        self,
        domain: str,
        query: str,
        include_trending: bool = True
    ) -> Dict[str, Any]:
        """Execute research for a specific domain, including trending content discovery."""
        try:
            # Use domain-specific agent if available
            if domain in self.domain_agents:
                agent = self.domain_agents[domain]
                agent_results = await agent.research(query)
                traditional_sources = agent_results.get('sources', [])
            else:
                # Fallback to web research tool for traditional sources
                traditional_sources = await self.web_research_tool.search_domain_sources(query, domain, max_sources=15)
            
            # Get enhanced sources that include trending content
            if include_trending:
                all_sources = await self.web_research_tool.enhanced_domain_search(
                    query=query,
                    domain=domain,
                    include_trending=True,
                    max_sources=25
                )
            else:
                all_sources = traditional_sources
            
            # Remove duplicates based on URL
            seen_urls = set()
            unique_sources = []
            for source in all_sources:
                url = source.get('url', '')
                # For trending topic ideas without URLs, use title as unique identifier
                unique_key = url if url else source.get('title', '')
                
                if unique_key and unique_key not in seen_urls:
                    seen_urls.add(unique_key)
                    unique_sources.append(source)
            
            # Separate traditional and trending for reporting
            traditional_count = len([s for s in unique_sources if s.get('source_type') not in ['trending_post', 'trending_topic_idea']])
            trending_count = len(unique_sources) - traditional_count
            
            return {
                'sources': unique_sources[:25],  # Limit to top 25 sources per domain
                'domain': domain,
                'sources_count': len(unique_sources),
                'traditional_sources_count': traditional_count,
                'trending_sources_count': trending_count
            }
                
        except Exception as e:
            logger.error(f"Domain research failed for {domain}: {e}")
            return {'sources': [], 'domain': domain, 'error': str(e)}
    
    async def _generate_topic_ideas(
        self,
        sources: List[Dict[str, Any]],
        domains: List[str],
        original_query: str
    ) -> List[TopicIdea]:
        """Generate topic ideas based on research sources."""
        
        # Prepare source summaries for the prompt
        source_summaries = []
        for i, source in enumerate(sources[:20]):  # Use top 20 sources
            summary = f"{i+1}. {source.get('title', 'Untitled')} - {source.get('summary', '')[:200]}"
            source_summaries.append(summary)
        
        system_prompt = research_agent_instruction
        
        human_prompt = f"""Original research query: "{original_query}"
Target domains: {', '.join(domains)}

Based on these research sources:
{chr(10).join(source_summaries)}

Generate 8-10 compelling topic ideas for long-form content. For each topic, provide:
1. A compelling, specific title
2. A detailed description (2-3 sentences)
3. The primary domain it belongs to
4. 3-5 target keywords
5. Why this topic would be valuable to readers

Format your response as JSON array with objects containing: title, description, domain, keywords, value_proposition.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            import json
            topic_data = json.loads(response.content)
            
            topics = []
            for item in topic_data:
                topic = TopicIdea(
                    title=item.get('title', ''),
                    description=item.get('description', ''),
                    domain=item.get('domain', domains[0]),
                    target_keywords=item.get('keywords', []),
                    relevance_score=0.8,  # Will be scored later
                    novelty_score=0.7,    # Will be scored later
                    seo_difficulty=0.5,   # Will be scored later
                    overall_score=0.7     # Will be calculated later
                )
                topics.append(topic)
            
            return topics
            
        except json.JSONDecodeError:
            logger.error("Failed to parse topic generation response")
            return []
    
    async def _score_topics(
        self,
        topics: List[TopicIdea],
        sources: List[Dict[str, Any]]
    ) -> List[TopicIdea]:
        """Score and rank topics based on multiple criteria."""
        
        system_prompt = research_agent_instruction
        
        topics_for_scoring = []
        for topic in topics:
            topics_for_scoring.append({
                'title': topic.title,
                'description': topic.description, 
                'domain': topic.domain,
                'keywords': topic.target_keywords
            })
        
        human_prompt = f"""Score these topic ideas:

{json.dumps(topics_for_scoring, indent=2)}

Return JSON array with objects containing: title, relevance_score, novelty_score, seo_difficulty, depth_potential.
Focus on practical, realistic scoring that reflects current content marketing conditions.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            import json
            scores_data = json.loads(response.content)
            
            # Apply scores to topics
            for i, topic in enumerate(topics):
                if i < len(scores_data):
                    scores = scores_data[i]
                    topic.relevance_score = scores.get('relevance_score', 0.7)
                    topic.novelty_score = scores.get('novelty_score', 0.7)
                    topic.seo_difficulty = scores.get('seo_difficulty', 0.5)
                    
                    # Calculate overall score (weighted)
                    topic.overall_score = (
                        topic.relevance_score * 0.4 +
                        topic.novelty_score * 0.3 +
                        (1.0 - topic.seo_difficulty) * 0.2 +  # Lower difficulty = higher score
                        scores.get('depth_potential', 0.7) * 0.1
                    )
            
            # Sort by overall score
            topics.sort(key=lambda t: t.overall_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Topic scoring failed: {e}")
            # Fallback: assign default scores
            for topic in topics:
                topic.overall_score = 0.7
        
        return topics
    
    async def _store_sources_in_milvus(
        self,
        sources: List[Dict[str, Any]],
        pipeline_run_id: str
    ) -> bool:
        """Store research sources in Milvus for future reference."""
        try:
            from app.services.vector_service import vector_service
            
            stored_count = 0
            for source in sources:
                # Convert source dict to required format
                success = await vector_service.insert_research_source(
                    source_id=str(uuid.uuid4()),
                    pipeline_run_id=pipeline_run_id,
                    url=source.get('url', ''),
                    title=source.get('title', ''),
                    domain=source.get('domain', ''),
                    source_type=source.get('source_type', 'unknown'),
                    summary=source.get('summary', ''),
                    credibility_score=source.get('credibility_score', 0.5),
                    relevance_score=source.get('relevance_score', 0.5)
                )
                
                if success:
                    stored_count += 1
            
            logger.info(f"Stored {stored_count}/{len(sources)} sources in Milvus")
            return stored_count > 0
            
        except Exception as e:
            logger.error(f"Failed to store sources in Milvus: {e}")
            return False
    
    async def _check_existing_research(
        self,
        query: str,
        domains: List[str]
    ) -> List[Dict[str, Any]]:
        """Check Milvus for existing research on similar topics."""
        try:
            from app.services.vector_service import vector_service
            
            # Search for similar sources across all domains
            all_existing_sources = []
            for domain in domains:
                existing_sources = await vector_service.search_similar_sources(
                    query_text=query,
                    domain=domain,
                    limit=10
                )
                all_existing_sources.extend(existing_sources)
            
            # Remove duplicates and sort by similarity
            seen_urls = set()
            unique_sources = []
            for source in all_existing_sources:
                url = source.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(source)
            
            # Sort by similarity score
            unique_sources.sort(key=lambda s: s.get('similarity_score', 0), reverse=True)
            
            logger.info(f"Found {len(unique_sources)} existing sources in Milvus for query: {query}")
            return unique_sources[:15]  # Limit to top 15
            
        except Exception as e:
            logger.error(f"Failed to check existing research: {e}")
            return []


class DomainResearchAgent:
    """Base class for domain-specific research agents."""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.web_research_tool = WebResearchTool()
    
    async def research(self, query: str, max_sources: int = 15) -> Dict[str, Any]:
        """Execute research for this domain."""
        sources = await self.web_research_tool.search_domain_sources(
            query, 
            self.domain_name, 
            max_sources
        )
        
        # Enhance sources with full content for top results
        enhanced_sources = []
        for source in sources[:10]:  # Enhance top 10 sources
            try:
                content_data = await self.web_research_tool.fetch_article_content(source['url'])
                if content_data.get('content'):
                    source.update(content_data)
                    # Update credibility score
                    source['credibility_score'] = await self.web_research_tool.validate_source_credibility(source)
                enhanced_sources.append(source)
            except Exception as e:
                logger.warning(f"Failed to enhance source {source.get('url', '')}: {e}")
                enhanced_sources.append(source)
        
        return {
            'sources': enhanced_sources,
            'domain': self.domain_name,
            'query': query
        }


class ITInsuranceResearchAgent(DomainResearchAgent):
    """Specialized agent for IT and insurance domain research."""
    
    def __init__(self):
        super().__init__("it_insurance")


class AIResearchAgent(DomainResearchAgent):
    """Specialized agent for AI domain research."""
    
    def __init__(self):
        super().__init__("ai")


class AgenticAIResearchAgent(DomainResearchAgent):
    """Specialized agent for agentic AI domain research."""
    
    def __init__(self):
        super().__init__("agentic_ai")


class AISoftwareEngineeringAgent(DomainResearchAgent):
    """Specialized agent for AI software engineering domain research."""
    
    def __init__(self):
        super().__init__("ai_software_engineering")