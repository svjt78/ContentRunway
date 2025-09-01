"""Content Formatter Agent - Formats content for different publishing platforms."""

from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
import re
import json

from ..state.pipeline_state import Draft, ChannelDrafts, ContentPipelineState

logger = logging.getLogger(__name__)


class ContentFormatterAgent:
    """Formats content for different publishing platforms with platform-specific optimization."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,  # Low temperature for consistent formatting
            max_tokens=4000
        )
        
        # Platform-specific formatting guidelines
        self.platform_guidelines = {
            'personal_blog': {
                'max_word_count': 3000,
                'ideal_word_count': 1800,
                'heading_style': 'title_case',
                'paragraph_length': 'varied',  # Mix of short and long
                'supports_code_blocks': True,
                'supports_images': True,
                'supports_lists': True,
                'formatting_style': 'rich_markdown',
                'seo_focus': 'high'
            }
        }
    
    async def execute(
        self,
        draft: Draft,
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """
        Format content for multiple publishing platforms.
        
        Args:
            draft: Content draft to format
            state: Current pipeline state
            
        Returns:
            Dictionary with platform-specific formatted content
        """
        logger.info("Starting multi-platform content formatting")
        
        try:
            # Determine target platforms
            target_platforms = self._determine_target_platforms(state)
            
            # Format for each platform
            channel_drafts = {}
            
            for platform in target_platforms:
                logger.info(f"Formatting content for {platform}")
                
                formatted_content = await self._format_for_platform(
                    draft, 
                    platform,
                    state
                )
                
                channel_drafts[platform] = formatted_content
            
            # Create ChannelDrafts object
            channel_drafts_obj = ChannelDrafts(**channel_drafts)
            
            logger.info(f"Content formatted for {len(target_platforms)} platforms")
            
            return {
                'channel_drafts': channel_drafts_obj,
                'platforms_formatted': target_platforms,
                'formatting_summary': self._generate_formatting_summary(channel_drafts, draft)
            }
            
        except Exception as e:
            logger.error(f"Content formatting failed: {e}")
            # Fallback: return original content in basic format
            fallback_drafts = ChannelDrafts(
                personal_blog={'content': draft.content, 'title': draft.title}
            )
            return {
                'channel_drafts': fallback_drafts,
                'platforms_formatted': ['personal_blog'],
                'error': str(e)
            }
    
    def _determine_target_platforms(self, state: ContentPipelineState) -> List[str]:
        """Determine which platforms to format content for based on state configuration."""
        
        # Check if platforms are specified in state
        configured_platforms = state.get('config_overrides', {}).get('target_platforms')
        
        if configured_platforms:
            return configured_platforms
        
        # Default platforms for Phase 1
        return ['personal_blog']
    
    async def _format_for_platform(
        self,
        draft: Draft,
        platform: str,
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Format content for a specific platform."""
        
        if platform not in self.platform_guidelines:
            logger.warning(f"Unknown platform: {platform}, using default formatting")
            platform = 'personal_blog'  # Fallback
        
        guidelines = self.platform_guidelines[platform]
        
        # Format for blog platform
        return await self._format_for_blog_platform(draft, platform, guidelines, state)
    
    async def _format_for_blog_platform(
        self,
        draft: Draft,
        platform: str,
        guidelines: Dict[str, Any],
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Format content for blog platforms (personal blog)."""
        
        system_prompt = f"""You are a content formatter specializing in {platform} optimization.

        Format the content according to these guidelines:
        - Target word count: {guidelines['ideal_word_count']} (max: {guidelines['max_word_count']})
        - Heading style: {guidelines['heading_style']}
        - Paragraph length: {guidelines['paragraph_length']}
        - Formatting style: {guidelines['formatting_style']}
        - SEO focus: {guidelines['seo_focus']}
        - Supports code blocks: {guidelines['supports_code_blocks']}
        
        Optimize for {platform}'s audience and format requirements while preserving content quality.
        """
        
        # Prepare content context
        current_word_count = draft.word_count
        needs_condensing = current_word_count > guidelines['max_word_count']
        
        human_prompt = f"""Format this content for {platform}:

        Original Title: {draft.title}
        Original Content: {draft.content}
        Current word count: {current_word_count}
        Target word count: {guidelines['ideal_word_count']}
        {'⚠️ Content needs condensing to meet platform limits' if needs_condensing else '✅ Content length is appropriate'}

        Return JSON with:
        {{
            "title": "platform-optimized title",
            "content": "formatted content",
            "meta_description": "SEO meta description",
            "tags": ["tag1", "tag2", "tag3"],
            "word_count": actual_word_count,
            "formatting_notes": "explanation of changes made"
        }}

        Ensure the formatted content maintains quality while fitting platform constraints.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            result = json.loads(response.content)
            
            # Add platform-specific metadata
            result['platform'] = platform
            result['guidelines_applied'] = guidelines
            result['seo_optimized'] = guidelines['seo_focus'] in ['high', 'medium']
            
            return result
            
        except Exception as e:
            logger.warning(f"Blog platform formatting failed for {platform}: {e}")
            # Fallback formatting
            return self._create_fallback_blog_format(draft, platform, guidelines)
    
    
    
    
    def _create_fallback_blog_format(
        self,
        draft: Draft,
        platform: str,
        guidelines: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create fallback formatting for blog platforms."""
        
        content = draft.content
        
        # Simple word count reduction if needed
        if len(content.split()) > guidelines['max_word_count']:
            words = content.split()
            content = ' '.join(words[:guidelines['max_word_count']])
            content += "\n\n[Content truncated for platform requirements]"
        
        return {
            'title': draft.title,
            'content': content,
            'meta_description': draft.meta_description or f"Learn about {draft.title}",
            'tags': draft.tags or [],
            'word_count': len(content.split()),
            'platform': platform,
            'formatting_notes': 'Fallback formatting applied due to processing error'
        }
    
    
    
    def _generate_formatting_summary(
        self,
        channel_drafts: Dict[str, Dict[str, Any]],
        original_draft: Draft
    ) -> Dict[str, Any]:
        """Generate summary of formatting changes across platforms."""
        
        summary = {
            'original_word_count': original_draft.word_count,
            'platforms_formatted': list(channel_drafts.keys()),
            'word_count_by_platform': {},
            'formatting_adaptations': [],
            'optimization_applied': []
        }
        
        for platform, formatted_content in channel_drafts.items():
            word_count = formatted_content.get('word_count', 0)
            summary['word_count_by_platform'][platform] = word_count
            
            # Track significant changes
            if word_count < original_draft.word_count * 0.7:
                summary['formatting_adaptations'].append(f"{platform}: Content significantly condensed")
            elif word_count > original_draft.word_count * 1.2:
                summary['formatting_adaptations'].append(f"{platform}: Content expanded with platform-specific elements")
            
            # Track optimizations
            if formatted_content.get('seo_optimized'):
                summary['optimization_applied'].append(f"{platform}: SEO optimization")
        
        return summary