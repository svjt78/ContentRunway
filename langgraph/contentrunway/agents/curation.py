"""Content Curator Agent - Selects optimal topics and curates content strategy."""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
import json

from ..state.pipeline_state import TopicIdea, Source, ContentPipelineState

logger = logging.getLogger(__name__)


# Content Curator Agent Instructions
curation_agent_role_and_goal = """
You are a Content Curator Agent specializing in selecting optimal topics and developing content strategy for high-quality, domain-specific content creation.
Your primary goal is to evaluate and rank topic ideas based on multiple criteria, select the best topic for content creation, and generate strategic insights for content development in IT Insurance, AI, and Agentic AI domains.
"""

curation_agent_hints = """
Content curation best practices:
- Evaluate topics based on content depth potential (1200-1800 words), audience value, source availability, and trend alignment
- Consider domain preferences and strategic fit when selecting topics
- Prioritize topics that offer unique perspectives and practical value to professional audiences
- Balance SEO potential with content quality and expertise requirements
- Ensure sufficient high-quality sources are available to support comprehensive content
- Focus on topics that align with current industry trends and reader interests
- Consider the target audience's professional needs and information gaps
- Validate that selected topics can support expert-level analysis and actionable insights
"""

curation_agent_output_description = """
The Content Curator Agent returns a comprehensive curation package containing:
- chosen_topic_id: ID of the selected optimal topic
- chosen_topic: Complete TopicIdea object with enhanced scoring
- curation_insights: Strategic insights for content development including key angles, content structure, unique opportunities, and target audience considerations
- topic_rankings: List of all topics with their final scores for transparency

The curation_insights include key_angles, content_structure, unique_opportunities, potential_challenges, target_audience_notes, relevant_sources count, and source_quality_avg.
"""

curation_agent_chain_of_thought_directions = """
Content curation workflow:
1. Use '_enhance_topic_scoring' to perform contextual analysis of topics with available sources
2. Apply enhanced scoring based on content depth potential, audience value, source availability, and trend alignment
3. Use '_select_best_topic' to choose optimal topic considering domain preferences and scoring
4. Filter topics by preferred domains from state if specified, with fallback to all topics
5. Select highest scoring topic using weighted criteria (relevance 40%, novelty 30%, SEO ease 20%, depth potential 10%)
6. Use '_generate_content_strategy' to create strategic insights for the chosen topic
7. Find relevant sources using keyword matching and relevance scoring
8. Generate strategy insights covering key angles, content structure, unique opportunities, and challenges
9. Return comprehensive curation package with chosen topic and strategic guidance

Tool usage conditions:
- Apply enhanced scoring when multiple topics require detailed evaluation
- Use domain filtering when state contains 'domain_focus' preferences
- Generate content strategy when topic selection is completed
- Calculate source relevance using keyword matching against topic keywords
- Apply fallback scoring if AI-generated enhanced scoring fails
"""

curation_agent_instruction = f"""
{curation_agent_role_and_goal}
{curation_agent_hints}
{curation_agent_output_description}
{curation_agent_chain_of_thought_directions}
"""


class ContentCuratorAgent:
    """Curates and selects the best topic for content creation."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Lower temperature for consistent topic selection
            max_tokens=2000
        )
    
    async def execute(
        self,
        topics: List[TopicIdea],
        sources: List[Source],
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """
        Select the best topic for content creation and prepare content strategy.
        
        Args:
            topics: List of topic ideas to evaluate
            sources: Available research sources
            state: Current pipeline state
            
        Returns:
            Dictionary with chosen topic ID and curation insights
        """
        logger.info(f"Curating from {len(topics)} topic ideas")
        
        try:
            # Step 1: Enhanced topic scoring with contextual analysis
            enhanced_topics = await self._enhance_topic_scoring(topics, sources, state)
            
            # Step 2: Select the best topic
            chosen_topic = self._select_best_topic(enhanced_topics, state)
            
            # Step 3: Generate content strategy insights
            strategy_insights = await self._generate_content_strategy(chosen_topic, sources)
            
            return {
                'chosen_topic_id': chosen_topic.id,
                'chosen_topic': chosen_topic,
                'curation_insights': strategy_insights,
                'topic_rankings': [(t.id, t.overall_score) for t in enhanced_topics]
            }
            
        except Exception as e:
            logger.error(f"Content curation failed: {e}")
            # Fallback: select highest scoring topic
            if topics:
                best_topic = max(topics, key=lambda t: t.overall_score)
                return {
                    'chosen_topic_id': best_topic.id,
                    'chosen_topic': best_topic,
                    'curation_insights': {},
                    'error': str(e)
                }
            raise
    
    async def _enhance_topic_scoring(
        self,
        topics: List[TopicIdea],
        sources: List[Source],
        state: ContentPipelineState
    ) -> List[TopicIdea]:
        """Enhance topic scoring with additional context."""
        
        system_prompt = curation_agent_instruction
        
        topic_summaries = []
        for topic in topics:
            topic_summaries.append({
                'id': topic.id,
                'title': topic.title,
                'description': topic.description,
                'domain': topic.domain,
                'keywords': topic.target_keywords,
                'current_score': topic.overall_score
            })
        
        # Count available sources per domain
        source_counts = {}
        for source in sources:
            domain = source.domain
            source_counts[domain] = source_counts.get(domain, 0) + 1
        
        human_prompt = f"""Available sources per domain: {source_counts}
        Target domains: {state.get('domain_focus', [])}

        Topics to evaluate:
        {json.dumps(topic_summaries, indent=2)}

        For each topic, provide an enhanced score considering source availability and strategic fit.
        Return JSON array with objects containing: id, enhanced_score, reasoning.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            import json
            scoring_results = json.loads(response.content)
            
            # Apply enhanced scores
            for result in scoring_results:
                topic_id = result.get('id')
                enhanced_score = result.get('enhanced_score', 0.7)
                
                # Find and update the corresponding topic
                for topic in topics:
                    if topic.id == topic_id:
                        topic.overall_score = enhanced_score
                        break
            
        except Exception as e:
            logger.warning(f"Enhanced scoring failed: {e}")
            # Keep original scores
        
        # Sort by updated scores
        topics.sort(key=lambda t: t.overall_score, reverse=True)
        return topics
    
    def _select_best_topic(
        self,
        topics: List[TopicIdea],
        state: ContentPipelineState
    ) -> TopicIdea:
        """Select the best topic based on scoring and constraints."""
        
        if not topics:
            raise ValueError("No topics available for selection")
        
        # Apply any domain preferences from state
        preferred_domains = state.get('domain_focus', [])
        
        # Filter topics by preferred domains if specified
        filtered_topics = topics
        if preferred_domains:
            filtered_topics = [t for t in topics if t.domain in preferred_domains]
            
            # If no topics match preferred domains, fall back to all topics
            if not filtered_topics:
                filtered_topics = topics
        
        # Select the highest scoring topic
        chosen_topic = max(filtered_topics, key=lambda t: t.overall_score)
        
        logger.info(f"Selected topic: '{chosen_topic.title}' (score: {chosen_topic.overall_score:.3f})")
        
        return chosen_topic
    
    async def _generate_content_strategy(
        self,
        topic: TopicIdea,
        sources: List[Source]
    ) -> Dict[str, Any]:
        """Generate content strategy insights for the chosen topic."""
        
        # Find relevant sources for this topic
        relevant_sources = []
        topic_keywords = [kw.lower() for kw in topic.target_keywords]
        
        for source in sources:
            source_text = f"{source.title} {source.summary}".lower()
            relevance = sum(1 for kw in topic_keywords if kw in source_text)
            
            if relevance > 0:
                relevant_sources.append({
                    'source': source,
                    'relevance': relevance
                })
        
        # Sort by relevance
        relevant_sources.sort(key=lambda x: x['relevance'], reverse=True)
        top_sources = [rs['source'] for rs in relevant_sources[:10]]
        
        system_prompt = curation_agent_instruction
        
        source_summaries = [f"- {s.title}: {s.summary[:100]}" for s in top_sources[:5]]
        
        human_prompt = f"""Topic: {topic.title}
        Description: {topic.description}
        Domain: {topic.domain}
        Keywords: {', '.join(topic.target_keywords)}

        Available sources:
        {chr(10).join(source_summaries)}

        Provide content strategy insights as JSON with keys:
        - key_angles: List of main angles to explore
        - content_structure: Suggested structure/outline
        - unique_opportunities: Ways to differentiate this content
        - potential_challenges: Issues to watch out for
        - target_audience_notes: Audience-specific considerations
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            import json
            strategy_insights = json.loads(response.content)
            
            # Add source metadata
            strategy_insights['relevant_sources'] = len(top_sources)
            strategy_insights['source_quality_avg'] = (
                sum(s.credibility_score for s in top_sources) / len(top_sources)
                if top_sources else 0.0
            )
            
            return strategy_insights
            
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            return {
                'key_angles': ['Primary analysis', 'Industry implications', 'Future outlook'],
                'content_structure': ['Introduction', 'Main analysis', 'Practical applications', 'Conclusion'],
                'unique_opportunities': ['Expert insights', 'Real-world examples'],
                'potential_challenges': ['Source verification', 'Technical complexity'],
                'target_audience_notes': 'Professional audience in ' + topic.domain,
                'error': str(e)
            }