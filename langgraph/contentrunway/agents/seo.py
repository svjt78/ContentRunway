"""SEO Strategist Agent - Develops SEO strategy and content outline."""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
import json
from collections import Counter

from ..state.pipeline_state import TopicIdea, Source, Outline, ContentPipelineState
from ..tools.langgraph_tools import LANGGRAPH_SEO_TOOLS, optimize_seo, analyze_content_quality

logger = logging.getLogger(__name__)


# SEO Strategist Agent Instructions
seo_agent_role_and_goal = """
You are an SEO Strategist Agent specializing in developing comprehensive SEO strategies and creating detailed content outlines for long-form professional content.
Your primary goal is to develop keyword strategies, create SEO-optimized content outlines, and provide specific optimization recommendations that will improve search rankings and user engagement for 1200-1800 word articles.
"""

seo_agent_hints = """
SEO strategy best practices:
- Develop one primary keyword with 3-5 secondary keywords and 5-10 long-tail keywords
- Create content outlines that naturally incorporate target keywords while maintaining readability
- Structure content with compelling introduction, 4-6 main sections, practical examples, and strong conclusion
- Consider search intent (informational/commercial/navigational) when developing strategy
- Optimize titles for 50-60 characters and meta descriptions for 150-155 characters
- Include opportunities for featured snippets, internal linking, and content optimization
- Balance keyword optimization with content quality and user value
- Focus on semantic keywords that add context and support the main keyword theme
"""

seo_agent_output_description = """
The SEO Strategist Agent returns a comprehensive SEO package containing:
- outline: Detailed Outline object with sections, estimated word counts, target audience, primary angle, and key takeaways
- keyword_strategy: Comprehensive keyword strategy with primary, secondary, long-tail, and semantic keywords
- seo_recommendations: Specific optimization recommendations including optimized title, meta description, header structure, internal linking opportunities, content tips, and featured snippet optimization

The outline includes sections with titles, subsections, estimated_words, and key_points, plus overall estimated_word_count and strategic elements.
"""

seo_agent_chain_of_thought_directions = """
SEO strategy development workflow:
1. Use '_develop_keyword_strategy' to create comprehensive keyword strategy based on topic and research sources
2. Extract common terms from source content using Counter for frequency analysis
3. Apply AI analysis to develop primary keyword, secondary keywords, long-tail keywords, and semantic keywords
4. Assess keyword difficulty and search intent for strategic planning
5. Use '_create_content_outline' to develop detailed, SEO-optimized content structure
6. Apply keyword strategy to outline creation with natural integration
7. Structure outline with introduction, 4-6 main sections, practical examples, and conclusion
8. Include specific sections for examples, case studies, and actionable insights
9. Use '_generate_seo_recommendations' to provide specific optimization guidance
10. Generate optimized title, meta description, header structure, and content optimization tips
11. Return comprehensive SEO package with outline and optimization strategy

Tool usage conditions:
- Use keyword strategy development when topic keywords need enhancement
- Apply content outline creation for all SEO strategy requests
- Generate SEO recommendations for every completed outline
- Use Counter from collections for term frequency analysis
- Apply fallback strategies if AI analysis fails JSON parsing
"""

seo_agent_instruction = f"""
{seo_agent_role_and_goal}
{seo_agent_hints}
{seo_agent_output_description}
{seo_agent_chain_of_thought_directions}
"""


class SEOStrategistAgent:
    """Develops SEO strategy and creates detailed content outline."""
    
    def __init__(self, model_name: str = "gpt-4", enable_tool_selection: bool = True):
        self.base_llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,  # Balanced for strategic planning
            max_tokens=3000
        )
        
        # Hybrid approach: bind SEO tools for LLM-driven optimization
        if enable_tool_selection:
            self.llm = self.base_llm.bind_tools(LANGGRAPH_SEO_TOOLS)
        else:
            self.llm = self.base_llm
    
    async def execute(
        self,
        topic: TopicIdea,
        sources: List[Source],
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """
        Develop SEO strategy and create detailed content outline.
        
        Args:
            topic: Selected topic for content creation
            sources: Available research sources
            state: Current pipeline state
            
        Returns:
            Dictionary with outline and SEO strategy
        """
        logger.info(f"Developing SEO strategy for: {topic.title}")
        
        try:
            # Step 1: Develop keyword strategy
            keyword_strategy = await self._develop_keyword_strategy(topic, sources)
            
            # Step 2: Create detailed outline
            outline = await self._create_content_outline(topic, sources, keyword_strategy, state)
            
            # Step 3: Generate SEO recommendations
            seo_recommendations = await self._generate_seo_recommendations(topic, outline)
            
            return {
                'outline': outline,
                'keyword_strategy': keyword_strategy,
                'seo_recommendations': seo_recommendations
            }
            
        except Exception as e:
            logger.error(f"SEO strategy development failed: {e}")
            # Create fallback outline
            fallback_outline = self._create_fallback_outline(topic, state)
            return {
                'outline': fallback_outline,
                'keyword_strategy': {'primary': topic.target_keywords[0] if topic.target_keywords else topic.title},
                'error': str(e)
            }
    
    async def _develop_keyword_strategy(
        self,
        topic: TopicIdea,
        sources: List[Source]
    ) -> Dict[str, Any]:
        """Develop comprehensive keyword strategy."""
        
        system_prompt = seo_agent_instruction
        
        # Extract common terms from sources for context
        source_terms = []
        for source in sources[:10]:
            terms = source.title.lower().split() + source.summary.lower().split()
            source_terms.extend([term for term in terms if len(term) > 3])
        
        # Get most common terms
        common_terms = [term for term, count in Counter(source_terms).most_common(20)]
        
        human_prompt = f"""Topic: {topic.title}
        Domain: {topic.domain}
        Current keywords: {', '.join(topic.target_keywords)}
        
        Common terms from research: {', '.join(common_terms)}
        
        Develop a keyword strategy for 1200-1800 word content targeting professional audience.
        
        Return JSON with:
        - primary_keyword: Single main keyword
        - secondary_keywords: Array of 3-5 supporting keywords
        - long_tail_keywords: Array of 5-10 specific phrases
        - semantic_keywords: Array of related terms
        - keyword_difficulty: Estimated difficulty (easy/medium/hard)
        - search_intent: Primary intent (informational/commercial/navigational)
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            import json
            keyword_strategy = json.loads(response.content)
            return keyword_strategy
            
        except Exception as e:
            logger.warning(f"Keyword strategy generation failed: {e}")
            # Fallback keyword strategy
            return {
                'primary_keyword': topic.target_keywords[0] if topic.target_keywords else topic.title.lower(),
                'secondary_keywords': topic.target_keywords[1:4] if len(topic.target_keywords) > 1 else [],
                'long_tail_keywords': [f"{topic.title.lower()} guide", f"how to {topic.title.lower()}"],
                'semantic_keywords': common_terms[:10] if common_terms else [],
                'keyword_difficulty': 'medium',
                'search_intent': 'informational'
            }
    
    async def _create_content_outline(
        self,
        topic: TopicIdea,
        sources: List[Source],
        keyword_strategy: Dict[str, Any],
        state: ContentPipelineState
    ) -> Outline:
        """Create detailed content outline with SEO optimization."""
        
        target_word_count = state.get('content_word_count_target', 1500)
        
        system_prompt = seo_agent_instruction
        
        # Prepare source context
        relevant_sources = []
        for source in sources[:15]:
            source_text = f"{source.title} {source.summary}".lower()
            keyword_matches = sum(1 for kw in keyword_strategy.get('secondary_keywords', []) 
                                 if kw.lower() in source_text)
            if keyword_matches > 0 or any(kw.lower() in source_text for kw in topic.target_keywords):
                relevant_sources.append(f"- {source.title}: {source.summary[:150]}")
        
        human_prompt = f"""Topic: {topic.title}
        Description: {topic.description}
        Domain: {topic.domain}
        Target word count: {target_word_count}
        
        Primary keyword: {keyword_strategy.get('primary_keyword', '')}
        Secondary keywords: {', '.join(keyword_strategy.get('secondary_keywords', []))}
        
        Relevant research sources:
        {chr(10).join(relevant_sources[:10])}
        
        Create a detailed outline as JSON with:
        - sections: Array of section objects with title, subsections, estimated_words, key_points
        - estimated_word_count: Total estimated words
        - target_audience: Specific audience description
        - primary_angle: Main angle/approach
        - key_takeaways: 3-5 main takeaways readers will gain
        - call_to_action: Suggested CTA for conclusion
        - internal_link_opportunities: Suggested topics for internal links
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            import json
            outline_data = json.loads(response.content)
            
            outline = Outline(
                sections=outline_data.get('sections', []),
                estimated_word_count=outline_data.get('estimated_word_count', target_word_count),
                target_audience=outline_data.get('target_audience', f'Professionals in {topic.domain}'),
                primary_angle=outline_data.get('primary_angle', topic.description),
                key_takeaways=outline_data.get('key_takeaways', []),
                call_to_action=outline_data.get('call_to_action'),
                primary_keyword=keyword_strategy.get('primary_keyword', ''),
                secondary_keywords=keyword_strategy.get('secondary_keywords', [])
            )
            
            return outline
            
        except Exception as e:
            logger.error(f"Outline creation failed: {e}")
            return self._create_fallback_outline(topic, state)
    
    def _create_fallback_outline(self, topic: TopicIdea, state: ContentPipelineState) -> Outline:
        """Create a basic fallback outline."""
        target_word_count = state.get('content_word_count_target', 1500)
        
        sections = [
            {
                'title': 'Introduction',
                'subsections': ['Background', 'Why This Matters'],
                'estimated_words': 200,
                'key_points': ['Set context', 'Preview main points']
            },
            {
                'title': 'Current Landscape',
                'subsections': ['Market Overview', 'Key Trends'],
                'estimated_words': 400,
                'key_points': ['Industry analysis', 'Trend identification']
            },
            {
                'title': 'Deep Dive Analysis',
                'subsections': ['Technical Details', 'Case Studies'],
                'estimated_words': 500,
                'key_points': ['Technical explanation', 'Real examples']
            },
            {
                'title': 'Practical Applications',
                'subsections': ['Implementation Guide', 'Best Practices'],
                'estimated_words': 300,
                'key_points': ['Actionable advice', 'Expert recommendations']
            },
            {
                'title': 'Conclusion',
                'subsections': ['Key Takeaways', 'Future Outlook'],
                'estimated_words': 200,
                'key_points': ['Summarize insights', 'Call to action']
            }
        ]
        
        return Outline(
            sections=sections,
            estimated_word_count=target_word_count,
            target_audience=f'Professionals in {topic.domain}',
            primary_angle=topic.description,
            key_takeaways=['Expert insights', 'Practical guidance', 'Industry trends'],
            primary_keyword=topic.target_keywords[0] if topic.target_keywords else topic.title.lower(),
            secondary_keywords=topic.target_keywords[1:4] if len(topic.target_keywords) > 1 else []
        )
    
    async def _generate_seo_recommendations(
        self,
        topic: TopicIdea,
        outline: Outline
    ) -> Dict[str, Any]:
        """Generate specific SEO recommendations for the content."""
        
        system_prompt = seo_agent_instruction
        
        human_prompt = f"""Topic: {topic.title}
        Primary keyword: {outline.primary_keyword}
        Secondary keywords: {', '.join(outline.secondary_keywords)}
        
        Content outline sections:
        {chr(10).join([f"- {section.get('title', 'Untitled')}" for section in outline.sections])}
        
        Provide SEO recommendations as JSON with:
        - optimized_title: SEO-optimized title (under 60 characters)
        - meta_description: Compelling meta description (under 160 characters)
        - header_structure: Recommended H1, H2, H3 hierarchy
        - internal_linking: Suggested internal link opportunities
        - content_tips: Specific optimization tips
        - featured_snippet_tips: How to optimize for featured snippets
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            import json
            seo_recommendations = json.loads(response.content)
            return seo_recommendations
            
        except Exception as e:
            logger.warning(f"SEO recommendations generation failed: {e}")
            return {
                'optimized_title': topic.title[:60],
                'meta_description': topic.description[:160],
                'header_structure': ['H1: Main title', 'H2: Major sections', 'H3: Subsections'],
                'internal_linking': ['Related articles', 'Category pages'],
                'content_tips': ['Use keywords naturally', 'Include examples', 'Add actionable advice'],
                'featured_snippet_tips': ['Use lists and tables', 'Answer questions directly']
            }