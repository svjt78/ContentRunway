"""Content Writer Agent - Generates high-quality content drafts with citations."""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
import re
import json
from datetime import datetime

from ..state.pipeline_state import Outline, Draft, Source, Citation, ContentPipelineState
from ..tools.langgraph_tools import (
    LANGGRAPH_WRITING_TOOLS,
    select_writing_style,
    analyze_content_quality,
    optimize_seo
)

logger = logging.getLogger(__name__)


# Content Writer Agent Instructions
writer_agent_role_and_goal = """
You are a Content Writer Agent specializing in generating comprehensive, high-quality content drafts with proper citations for professional audiences in IT Insurance, AI, and Agentic AI domains.
Your primary goal is to create engaging, informative content (1200-1800 words) that provides actionable insights, maintains professional tone, incorporates relevant citations naturally, and follows logical content structure based on detailed outlines.
"""

writer_agent_hints = """
Content writing best practices:
- Write for professional audiences who want in-depth analysis and practical value
- Use short paragraphs (2-4 sentences) for readability and engagement
- Include transition sentences between major points for logical flow
- Support claims with credible sources and data using [Citation X] format
- Provide specific examples, case studies, and actionable insights
- Maintain consistent professional tone while being engaging and accessible
- Focus on the target audience and primary angle specified in the outline
- Structure introductions with hooks, context, and preview; conclusions with summaries and calls-to-action
- Aim for approximately 200 words per minute reading time (average reading speed)
- Include practical applications and real-world examples to enhance value
"""

writer_agent_output_description = """
The Content Writer Agent returns a comprehensive content package containing:
- draft: Complete Draft object with title, content, citations, word count, reading time, and metadata
- writing_metadata: Detailed metadata including sections_generated, citations_used, target_keywords_covered, and content_quality_indicators

The draft includes title, subtitle, abstract, content with proper section headers, citations list, word_count, reading_time_minutes, meta_description, keywords, and tags.
Writing metadata provides insights into content structure, keyword coverage, and quality indicators for assessment.
"""

writer_agent_chain_of_thought_directions = """
Content writing workflow:
1. Use '_prepare_writing_context' to organize sources by credibility and relevance, create citation mapping
2. Sort sources using weighted scoring (credibility_score * 0.4 + relevance_score * 0.6)
3. Create Citation objects for top 15 sources with incremental numbering
4. Use '_generate_content_sections' to write content for each outline section
5. Apply '_write_section' for each section with appropriate context (introduction, main content, conclusion)
6. Use '_find_relevant_sources' to match sources to section content using keyword analysis
7. Apply section-specific writing approaches: introductions include hooks and previews, main sections include analysis and examples, conclusions include summaries and CTAs
8. Use '_extract_citations' to process [Citation X] references and validate against available sources
9. Use '_generate_title_and_metadata' to create compelling titles and SEO metadata
10. Apply '_assemble_draft' to combine all components into final Draft object
11. Use '_calculate_metrics' to determine reading time and content metrics
12. Return complete content package with draft and comprehensive metadata

Tool usage conditions:
- Apply source enhancement with 'fetch_article_content' for top 10 sources
- Use 'validate_source_credibility' to update credibility scores
- Generate section content when outline sections are provided
- Apply citation extraction using regex pattern \[Citation (\d+)\]
- Calculate reading time using 200 words per minute standard
- Generate metadata when title and SEO optimization are required
"""

writer_agent_instruction = f"""
{writer_agent_role_and_goal}
{writer_agent_hints}
{writer_agent_output_description}
{writer_agent_chain_of_thought_directions}
"""


class ContentWriterAgent:
    """Generates comprehensive content drafts based on outlines and sources."""
    
    def __init__(self, model_name: str = "gpt-4", enable_tool_selection: bool = True):
        self.base_llm = ChatOpenAI(
            model=model_name,
            temperature=0.4,  # Balanced creativity for engaging writing
            max_tokens=4000
        )
        
        # Hybrid approach: bind tools for LLM-driven decisions
        if enable_tool_selection:
            self.llm = self.base_llm.bind_tools(LANGGRAPH_WRITING_TOOLS)
        else:
            self.llm = self.base_llm
    
    async def execute(
        self,
        outline: Outline,
        sources: List[Source],
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """
        Generate a complete content draft based on the outline and sources.
        
        Args:
            outline: Content outline with structure and strategy
            sources: Research sources for citations
            state: Current pipeline state
            
        Returns:
            Dictionary with the generated draft and metadata
        """
        logger.info(f"Starting content writing for {outline.estimated_word_count} word article")
        
        try:
            # Step 1: Let LLM select writing style
            style_decision = await self._select_writing_style(outline, state)
            
            # Step 2: Prepare writing context
            writing_context = self._prepare_writing_context(outline, sources, state)
            writing_context['style_config'] = style_decision
            
            # Step 3: Generate the main content
            content_sections = await self._generate_content_sections(outline, writing_context)
            
            # Step 4: Create title and metadata
            title_metadata = await self._generate_title_and_metadata(outline, content_sections)
            
            # Step 5: Assemble the final draft
            draft = self._assemble_draft(
                title_metadata, 
                content_sections, 
                outline, 
                writing_context['citations']
            )
            
            # Step 6: Let LLM analyze and optimize content
            optimization_results = await self._optimize_content(draft, outline)
            if optimization_results.get('status') == 'success':
                draft = self._apply_optimizations(draft, optimization_results['data'])
            
            # Step 7: Calculate reading time and metrics
            draft = self._calculate_metrics(draft)
            
            logger.info(f"Content draft completed: {draft.word_count} words, {draft.reading_time_minutes}min read")
            
            return {
                'draft': draft,
                'writing_metadata': {
                    'sections_generated': len(content_sections),
                    'citations_used': len(draft.citations),
                    'target_keywords_covered': self._count_keyword_coverage(draft, outline),
                    'content_quality_indicators': self._assess_content_quality(draft, outline),
                    'style_decisions': style_decision,
                    'optimization_applied': optimization_results.get('status') == 'success'
                }
            }
            
        except Exception as e:
            logger.error(f"Content writing failed: {e}")
            raise
    
    def _prepare_writing_context(
        self,
        outline: Outline,
        sources: List[Source],
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Prepare context and citations for writing."""
        
        # Organize sources by relevance and credibility
        prioritized_sources = sorted(
            sources,
            key=lambda s: (s.credibility_score * 0.4 + s.relevance_score * 0.6),
            reverse=True
        )
        
        # Prepare citation mapping
        citations = []
        citation_counter = 1
        
        # Create citations for high-quality sources
        for source in prioritized_sources[:15]:  # Limit to top 15 sources
            citation = Citation(
                number=citation_counter,
                source=source,
                quote_text="",  # Will be filled during writing
                context="",     # Will be filled during writing
                citation_type="reference"
            )
            citations.append(citation)
            citation_counter += 1
        
        return {
            'prioritized_sources': prioritized_sources,
            'citations': citations,
            'domain_focus': state.get('domain_focus', []),
            'target_audience': outline.target_audience,
            'primary_angle': outline.primary_angle
        }
    
    async def _generate_content_sections(
        self,
        outline: Outline,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate content for each section of the outline."""
        
        sections = []
        
        for i, section_data in enumerate(outline.sections):
            section_title = section_data.get('title', f'Section {i+1}')
            section_points = section_data.get('points', [])
            section_word_target = section_data.get('word_count', 200)
            
            logger.info(f"Writing section: {section_title}")
            
            # Generate section content
            section_content = await self._write_section(
                section_title,
                section_points,
                section_word_target,
                context,
                is_introduction=(i == 0),
                is_conclusion=(i == len(outline.sections) - 1)
            )
            
            sections.append({
                'title': section_title,
                'content': section_content['text'],
                'citations_used': section_content['citations'],
                'word_count': len(section_content['text'].split())
            })
        
        return sections
    
    async def _write_section(
        self,
        section_title: str,
        section_points: List[str],
        word_target: int,
        context: Dict[str, Any],
        is_introduction: bool = False,
        is_conclusion: bool = False
    ) -> Dict[str, Any]:
        """Write a single section with proper citations."""
        
        system_prompt = writer_agent_instruction
        
        # Prepare relevant sources for this section
        relevant_sources = self._find_relevant_sources(section_points, context['prioritized_sources'])
        source_summaries = [
            f"Source {i+1}: {source.title} - {source.summary[:150]}"
            for i, source in enumerate(relevant_sources[:5])
        ]
        
        section_type_guidance = ""
        if is_introduction:
            section_type_guidance = """This is the introduction section. Include:
            - Hook to grab attention
            - Brief context/background
            - Clear preview of what readers will learn
            - Thesis or main argument"""
        elif is_conclusion:
            section_type_guidance = """This is the conclusion section. Include:
            - Summary of key insights
            - Practical implications
            - Call to action or next steps
            - Forward-looking perspective"""
        else:
            section_type_guidance = """This is a main content section. Include:
            - Clear topic introduction
            - Detailed analysis with evidence
            - Examples and practical applications
            - Smooth transition to next point"""
        
        human_prompt = f"""Section: {section_title}

        Key points to cover:
        {chr(10).join(f'- {point}' for point in section_points)}

        {section_type_guidance}

        Available sources:
        {chr(10).join(source_summaries)}

        Write the section content. Use [Citation X] format when referencing sources, where X is the source number.
        Aim for approximately {word_target} words.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            section_text = response.content
            
            # Extract and process citations
            citations_used = self._extract_citations(section_text, relevant_sources)
            
            return {
                'text': section_text,
                'citations': citations_used
            }
            
        except Exception as e:
            logger.error(f"Section writing failed for '{section_title}': {e}")
            # Fallback content
            fallback_content = f"""## {section_title}

            This section covers {', '.join(section_points[:2]) if section_points else 'important aspects of the topic'}.
            
            [Content generation in progress - this section needs manual completion due to technical issues.]
            """
            return {'text': fallback_content, 'citations': []}
    
    def _find_relevant_sources(self, section_points: List[str], sources: List[Source]) -> List[Source]:
        """Find sources most relevant to the section points."""
        
        # Create search terms from section points
        search_terms = []
        for point in section_points:
            # Extract key terms (simple approach)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', point.lower())
            search_terms.extend(words)
        
        # Score sources by relevance
        scored_sources = []
        for source in sources:
            source_text = f"{source.title} {source.summary}".lower()
            relevance_score = sum(1 for term in search_terms if term in source_text)
            
            if relevance_score > 0:
                scored_sources.append((source, relevance_score))
        
        # Return top sources
        scored_sources.sort(key=lambda x: x[1], reverse=True)
        return [source for source, score in scored_sources[:8]]
    
    def _extract_citations(self, text: str, sources: List[Source]) -> List[int]:
        """Extract citation numbers from text and validate against sources."""
        citation_pattern = r'\[Citation (\d+)\]'
        citation_matches = re.findall(citation_pattern, text)
        
        # Convert to integers and validate
        citation_numbers = []
        for match in citation_matches:
            try:
                num = int(match)
                if 1 <= num <= len(sources):
                    citation_numbers.append(num)
            except ValueError:
                continue
        
        return list(set(citation_numbers))  # Remove duplicates
    
    async def _generate_title_and_metadata(
        self,
        outline: Outline,
        sections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate compelling title and SEO metadata."""
        
        system_prompt = writer_agent_instruction
        
        # Summarize content for context
        content_summary = "\n".join([
            f"Section: {section['title']} ({section['word_count']} words)"
            for section in sections[:3]  # First 3 sections for context
        ])
        
        human_prompt = f"""Content outline:
        Primary keyword: {outline.primary_keyword}
        Secondary keywords: {', '.join(outline.secondary_keywords[:3])}
        Target audience: {outline.target_audience}
        Primary angle: {outline.primary_angle}
        Key takeaways: {', '.join(outline.key_takeaways[:3])}

        Content structure:
        {content_summary}

        Generate:
        1. A compelling main title
        2. An optional subtitle
        3. A meta description for SEO
        4. 3 alternative titles

        Return as JSON with keys: title, subtitle, meta_description, alternative_titles
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            import json
            metadata = json.loads(response.content)
            return metadata
            
        except Exception as e:
            logger.error(f"Title generation failed: {e}")
            # Fallback
            return {
                'title': f"The Complete Guide to {outline.primary_keyword}",
                'subtitle': None,
                'meta_description': f"Learn everything about {outline.primary_keyword} in this comprehensive guide for {outline.target_audience}.",
                'alternative_titles': [
                    f"Understanding {outline.primary_keyword}: A Professional Guide",
                    f"Mastering {outline.primary_keyword} for {outline.target_audience}",
                    f"{outline.primary_keyword} Explained: Best Practices and Insights"
                ]
            }
    
    def _assemble_draft(
        self,
        title_metadata: Dict[str, Any],
        sections: List[Dict[str, Any]],
        outline: Outline,
        citations: List[Citation]
    ) -> Draft:
        """Assemble the complete draft from all components."""
        
        # Combine all section content
        full_content = []
        all_citations_used = []
        
        for section in sections:
            if section['title'] and not section['content'].startswith(f"## {section['title']}"):
                full_content.append(f"## {section['title']}\n")
            full_content.append(section['content'])
            full_content.append("\n")  # Section separator
            all_citations_used.extend(section['citations_used'])
        
        content_text = "\n".join(full_content).strip()
        
        # Process citations
        used_citation_numbers = list(set(all_citations_used))
        final_citations = []
        
        for citation_num in used_citation_numbers:
            if citation_num <= len(citations):
                citation = citations[citation_num - 1]
                citation.number = citation_num
                final_citations.append(citation)
        
        # Extract keywords from content
        content_keywords = self._extract_keywords(content_text, outline)
        
        return Draft(
            title=title_metadata['title'],
            subtitle=title_metadata.get('subtitle'),
            content=content_text,
            citations=final_citations,
            word_count=len(content_text.split()),
            reading_time_minutes=0,  # Will be calculated
            meta_description=title_metadata.get('meta_description'),
            keywords=content_keywords,
            tags=outline.secondary_keywords[:5]
        )
    
    def _extract_keywords(self, content: str, outline: Outline) -> List[str]:
        """Extract relevant keywords from the content."""
        keywords = [outline.primary_keyword]
        keywords.extend(outline.secondary_keywords[:5])
        
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add high-frequency domain-relevant terms
        common_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        for term, freq in common_terms:
            if freq >= 3 and term not in [kw.lower() for kw in keywords]:
                keywords.append(term)
        
        return keywords[:15]
    
    def _calculate_metrics(self, draft: Draft) -> Draft:
        """Calculate reading time and other content metrics."""
        
        # Calculate reading time (average 200 words per minute)
        words_per_minute = 200
        reading_time = max(1, round(draft.word_count / words_per_minute))
        draft.reading_time_minutes = reading_time
        
        return draft
    
    def _count_keyword_coverage(self, draft: Draft, outline: Outline) -> Dict[str, int]:
        """Count how many target keywords appear in the content."""
        content_lower = draft.content.lower()
        
        coverage = {}
        coverage['primary'] = content_lower.count(outline.primary_keyword.lower())
        
        for keyword in outline.secondary_keywords:
            coverage[keyword] = content_lower.count(keyword.lower())
        
        return coverage
    
    def _assess_content_quality(self, draft: Draft, outline: Outline) -> Dict[str, Any]:
        """Assess basic content quality indicators."""
        
        content_lines = draft.content.split('\n')
        paragraphs = [line.strip() for line in content_lines if line.strip() and not line.startswith('#')]
        
        return {
            'paragraph_count': len(paragraphs),
            'avg_paragraph_length': sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
            'sections_count': len([line for line in content_lines if line.startswith('##')]),
            'citations_count': len(draft.citations),
            'word_count_vs_target': draft.word_count / outline.estimated_word_count if outline.estimated_word_count else 1.0,
            'primary_keyword_density': draft.content.lower().count(outline.primary_keyword.lower()) / draft.word_count * 100
        }
    
    async def _select_writing_style(
        self, 
        outline: Outline, 
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Let LLM select appropriate writing style based on context."""
        
        system_prompt = writer_agent_instruction
        
        domain_focus = state.get('domain_focus', ['general'])
        content_type = "technical_guide" if any(d in ["ai", "agentic_ai"] for d in domain_focus) else "article"
        
        human_prompt = f"""Based on this content outline, select the most appropriate writing style:
        
        Topic: {outline.primary_keyword}
        Target Audience: {outline.target_audience}
        Domain: {', '.join(domain_focus)}
        Estimated Length: {outline.estimated_word_count} words
        Primary Angle: {outline.primary_angle}
        
        Choose the writing style by calling the select_writing_style tool with appropriate parameters.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Check if LLM used tool
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Tool was called, extract result
                tool_call = response.tool_calls[0]
                if tool_call['name'] == 'select_writing_style':
                    return tool_call['result']
            
            # Fallback if no tool was used
            return select_writing_style(
                content_type=content_type,
                audience=outline.target_audience,
                tone="professional"
            )
            
        except Exception as e:
            logger.error(f"Writing style selection failed: {e}")
            # Default fallback
            return {
                "status": "success",
                "data": {
                    "style_config": {
                        "paragraph_length": "medium",
                        "technical_depth": "medium",
                        "examples": "case_studies",
                        "structure": "problem_solution"
                    }
                }
            }
    
    async def _optimize_content(
        self, 
        draft: Draft, 
        outline: Outline
    ) -> Dict[str, Any]:
        """Let LLM analyze and optimize content quality."""
        
        system_prompt = writer_agent_instruction
        
        human_prompt = f"""Analyze this content and optimize it if needed:
        
        Title: {draft.title}
        Word Count: {draft.word_count}
        Primary Keyword: {outline.primary_keyword}
        Secondary Keywords: {', '.join(outline.secondary_keywords[:3])}
        
        Content preview: {draft.content[:500]}...
        
        Use the appropriate analysis tools to check content quality and SEO optimization.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Check if LLM used optimization tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Return the tool results
                return response.tool_calls[0].get('result', {'status': 'success', 'data': {}})
            
            # No tools used, return success
            return {'status': 'success', 'data': {}}
            
        except Exception as e:
            logger.error(f"Content optimization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _apply_optimizations(
        self, 
        draft: Draft, 
        optimization_data: Dict[str, Any]
    ) -> Draft:
        """Apply optimization recommendations to the draft."""
        
        # Apply any optimization suggestions
        if 'recommendations' in optimization_data:
            recommendations = optimization_data['recommendations']
            
            # Apply title optimization if suggested
            if 'title_suggestions' in recommendations and recommendations['title_suggestions']:
                draft.title = recommendations['title_suggestions'][0]
            
            # Apply meta description optimization
            if 'meta_description' in recommendations:
                draft.meta_description = recommendations['meta_description']
        
        return draft