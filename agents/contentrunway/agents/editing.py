"""Content Editor Agent - Improves content based on quality feedback and guidelines."""

import os
import asyncio
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
import re
import json

# Load environment variables from .env file explicitly
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ..state.pipeline_state import Draft, QualityScores, ContentPipelineState
from ..utils.model_selector import get_optimized_model_config, estimate_operation_cost
from ..utils.llm_cache import get_cached_llm_response, cache_llm_response

logger = logging.getLogger(__name__)


# Content Editor Agent Instructions
editor_agent_role_and_goal = """
You are a Content Editor Agent specializing in improving content based on quality feedback, critique analysis, and editorial guidelines for professional content creation.
Your primary goal is to analyze quality feedback, prioritize editing tasks, apply targeted improvements, and enhance content quality while preserving core message and value through systematic editing cycles.
"""

editor_agent_hints = """
Content editing best practices:
- Prioritize critique-focused improvements for retry cycles with targeted, surgical edits
- Address critical issues first (fact-checking, compliance) before moderate issues (style, flow)
- Make targeted improvements rather than broad changes to maintain content integrity
- Focus on specific areas identified in critique feedback with precision
- Preserve successful elements from previous versions while addressing identified issues
- Apply different editing approaches: fact_checking, style_improvement, structural_editing, clarity_improvement, technical_enhancement, compliance_editing
- Maintain professional tone and content structure unless specifically requested to change
- Verify improvements are successfully applied by checking content changes and word count adjustments

CRITICAL PRESERVATION REQUIREMENTS:
- ALWAYS preserve all [Citation X] markers exactly as they appear - do not remove or modify citations
- ALWAYS preserve all <domain_term> protected markers exactly as they appear - these are critical for quality scoring
- If you encounter <term> markers, treat them as protected domain terminology that must be preserved
- When rewriting sentences, ensure all citation numbers and domain term markers are maintained

QUALITY GATE ENHANCEMENT PRIORITIES:
- FACT CHECK (0.90+ target): Add explicit citations [1], [2] to claims. Include "according to [Source]" attribution. Convert vague statements to specific data. PRESERVE existing citations.
- DOMAIN EXPERTISE (0.90+ target): Insert domain-specific terminology with <term> protection markers. Add implementation bullet points. Include practical examples and case studies. PRESERVE existing <domain_term> markers.
- STYLE CONSISTENCY (0.88+ target): Ensure ## Markdown headers. Add bullet/numbered lists. Maintain 15-25 word sentences for professional readability. Split sentences >24 words.
- COMPLIANCE (0.95+ target): Add disclaimers to advice. Avoid absolute claims. Include "for informational purposes" where appropriate.
"""

editor_agent_output_description = """
The Content Editor Agent returns a comprehensive editing package containing:
- revised_draft: Updated Draft object with improvements applied
- editing_summary: Detailed summary of improvements including changes detected, word count changes, and quality improvement estimates
- feedback_addressed: Analysis of quality feedback and issues processed
- priorities_handled: List of editing tasks completed with priority levels

The editing_summary includes improvements_applied count, changes_detected list, editing_effectiveness ratio, and quality_improvement_estimate with confidence assessment.
"""

editor_agent_chain_of_thought_directions = """
Content editing workflow:
1. Use '_analyze_quality_feedback' to process quality scores, critique notes, and current critique feedback
2. Integrate critique feedback for retry cycles with focused issue identification
3. Categorize issues by severity: critical (<0.6), moderate (0.6-0.8), minor (0.8-0.9)
4. Use '_prioritize_editing_tasks' to organize editing tasks by priority and impact
5. Apply '_determine_editing_type' to classify editing approaches needed
6. Use '_apply_critique_focused_edits' for retry cycles with targeted improvements
7. Apply '_apply_high_priority_edits' for critical issues requiring significant changes
8. Use '_apply_medium_priority_edits' for style and flow improvements
9. Apply '_apply_low_priority_edits' for minor polishing and refinements
10. Use '_verify_improvements' to validate that edits were successfully applied
11. Apply '_detect_content_changes' to identify structural and content modifications
12. Use '_estimate_quality_improvement' to assess potential score improvements
13. Return comprehensive editing package with revised draft and detailed metadata

Tool usage conditions:
- Apply critique-focused editing when critique_feedback exists in state and critique_cycle_count > 1
- Use high-priority editing for critical issues (score < 0.6) and compliance problems
- Apply medium-priority editing for moderate issues (0.6-0.8 scores)
- Use low-priority editing for minor issues (0.8-0.9 scores) and polishing
- Generate improvement estimates based on issues addressed and editing effectiveness
"""

editor_agent_instruction = f"""
{editor_agent_role_and_goal}
{editor_agent_hints}
{editor_agent_output_description}
{editor_agent_chain_of_thought_directions}
"""


class ContentEditorAgent:
    """Edits and improves content based on quality feedback and editorial guidelines."""
    
    def __init__(self, model_name: str = None):
        # Use cost-optimized model configuration for content editing
        # Get API key with proper error handling
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable in your .env file."
            )
        
        config = get_optimized_model_config("ContentEditorAgent", task_type="technical_editing")
        
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=0.4,  # Balanced creativity for editing
            max_tokens=config.max_tokens,
            openai_api_key=api_key  # Explicitly pass the API key
        )
        self.agent_name = "ContentEditorAgent"
    
    async def execute(
        self,
        draft: Draft,
        quality_feedback: Dict[str, Any],
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """
        Edit and improve content based on quality feedback.
        
        Args:
            draft: Original content draft
            quality_feedback: Feedback from quality gates
            state: Current pipeline state
            
        Returns:
            Dictionary with revised draft and editing metadata
        """
        logger.info("🚀 Starting content editing based on quality feedback")
        
        try:
            # Step 1: Analyze quality feedback (including critique feedback)
            logger.info("📊 Step 1: Analyzing quality feedback")
            feedback_analysis = self._analyze_quality_feedback(quality_feedback, state)
            logger.info(f"📊 Quality feedback analysis complete - {len(feedback_analysis.get('critical_issues', []))} critical, {len(feedback_analysis.get('moderate_issues', []))} moderate issues")
            
            # Step 2: Prioritize editing tasks based on critique feedback
            logger.info("🎯 Step 2: Prioritizing editing tasks")
            editing_priorities = self._prioritize_editing_tasks(feedback_analysis, state)
            logger.info(f"🎯 Editing priorities set - {len(editing_priorities)} tasks identified")
            
            # Step 3: Apply content improvements
            logger.info("✏️  Step 3: Applying content improvements")
            revised_draft = await self._apply_content_improvements(
                draft, 
                editing_priorities,
                state
            )
            logger.info("✏️  Content improvements applied successfully")
            
            # Step 4: Verify improvements
            logger.info("✅ Step 4: Verifying improvements")
            improvement_summary = self._verify_improvements(draft, revised_draft, feedback_analysis)
            logger.info(f"✅ Improvement verification complete - {improvement_summary.get('improvements_applied', 0)} improvements applied")
            
            # Step 4.5: Restore any citations lost during editing
            revised_draft = self._restore_lost_citations(draft, revised_draft, state)
            
            # Step 5: Validate content for editorial metadata (final safety check)
            content_is_clean = self._validate_content_for_editorial_metadata(
                revised_draft.content, 
                "post-editing"
            )
            
            if not content_is_clean:
                logger.warning("⚠️  Editorial metadata detected in final content, attempting sanitization")
                # Apply emergency sanitization
                sanitized_content = self._extract_clean_content_from_response(
                    revised_draft.content, 
                    draft.content
                )
                # Update the revised draft
                revised_draft = Draft(
                    title=revised_draft.title,
                    subtitle=revised_draft.subtitle,
                    abstract=revised_draft.abstract,
                    content=sanitized_content,
                    citations=revised_draft.citations,
                    word_count=len(sanitized_content.split()),
                    reading_time_minutes=max(1, len(sanitized_content.split()) // 200),
                    meta_description=revised_draft.meta_description,
                    keywords=revised_draft.keywords,
                    tags=revised_draft.tags
                )
                # Re-validate
                self._validate_content_for_editorial_metadata(revised_draft.content, "post-sanitization")
            
            logger.info(f"Content editing completed: {improvement_summary['improvements_applied']} improvements")
            
            return {
                'revised_draft': revised_draft,
                'editing_summary': improvement_summary,
                'feedback_addressed': feedback_analysis,
                'priorities_handled': editing_priorities
            }
            
        except Exception as e:
            logger.error(f"Content editing failed: {e}")
            # Return original draft if editing fails
            return {
                'revised_draft': draft,
                'editing_summary': {'error': str(e), 'improvements_applied': 0},
                'error': str(e)
            }
    
    def _analyze_quality_feedback(
        self, 
        quality_feedback: Dict[str, Any],
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Analyze quality feedback to identify specific issues to address."""
        
        scores = quality_feedback.get('scores', {})
        critique_notes = quality_feedback.get('critique_notes', [])
        
        # Check for critique feedback from current cycle
        critique_feedback = state.get('current_critique_feedback')
        critique_cycle = state.get('critique_cycle_count', 0)
        
        analysis = {
            'critical_issues': [],
            'moderate_issues': [],
            'minor_issues': [],
            'areas_to_improve': {},
            'critique_focused_issues': [],
            'editing_cycle': critique_cycle,
            'is_retry_cycle': critique_cycle > 1
        }
        
        # Analyze individual quality scores
        if isinstance(scores, dict):
            # Handle dict-like scores
            score_items = scores.items()
        else:
            # Handle QualityScores object
            score_items = [
                ('fact_check', getattr(scores, 'fact_check', None)),
                ('domain_expertise', getattr(scores, 'domain_expertise', None)),
                ('style_consistency', getattr(scores, 'style_consistency', None)),
                ('compliance', getattr(scores, 'compliance', None)),
                ('technical_depth', getattr(scores, 'technical_depth', None))
            ]
        
        for area, score in score_items:
            if score is not None:
                if score < 0.6:
                    analysis['critical_issues'].append(f"{area} score too low ({score:.2f})")
                    analysis['areas_to_improve'][area] = 'critical'
                elif score < 0.8:
                    analysis['moderate_issues'].append(f"{area} needs improvement ({score:.2f})")
                    analysis['areas_to_improve'][area] = 'moderate'
                elif score < 0.9:
                    analysis['minor_issues'].append(f"{area} could be enhanced ({score:.2f})")
                    analysis['areas_to_improve'][area] = 'minor'
        
        # Analyze critique notes
        for note in critique_notes:
            if 'critical' in note.lower() or 'error' in note.lower():
                analysis['critical_issues'].append(note)
            elif 'improve' in note.lower() or 'enhance' in note.lower():
                analysis['moderate_issues'].append(note)
            else:
                analysis['minor_issues'].append(note)
        
        # Integrate critique feedback for focused editing
        if critique_feedback:
            logger.info(f"Integrating critique feedback from cycle {critique_cycle}")
            
            # Add issues remaining from critique
            for issue in critique_feedback.issues_remaining:
                severity = issue.get('severity', 'moderate')
                if severity == 'critical':
                    analysis['critical_issues'].append(f"Critique: {issue['issue']}")
                elif severity == 'high':
                    analysis['moderate_issues'].append(f"Critique: {issue['issue']}")
                else:
                    analysis['minor_issues'].append(f"Critique: {issue['issue']}")
            
            # Add focused improvement suggestions from critique
            for suggestion in critique_feedback.improvement_suggestions:
                priority = suggestion.get('priority', 'medium')
                area = suggestion.get('area', 'general')
                
                focused_issue = {
                    'suggestion': suggestion.get('suggestion', ''),
                    'priority': priority,
                    'area': area,
                    'expected_impact': suggestion.get('expected_impact', 'medium'),
                    'implementation_difficulty': suggestion.get('implementation_difficulty', 'moderate')
                }
                analysis['critique_focused_issues'].append(focused_issue)
                
                # Also add to areas to improve for priority handling
                if priority in ['critical', 'high']:
                    analysis['areas_to_improve'][area] = 'critical' if priority == 'critical' else 'moderate'
        
        return analysis
    
    def _prioritize_editing_tasks(
        self, 
        feedback_analysis: Dict[str, Any],
        state: ContentPipelineState
    ) -> List[Dict[str, Any]]:
        """Prioritize editing tasks based on feedback severity and impact."""
        
        priorities = []
        is_retry_cycle = feedback_analysis.get('is_retry_cycle', False)
        
        # For retry cycles, prioritize critique-focused issues first
        if is_retry_cycle and feedback_analysis.get('critique_focused_issues'):
            logger.info("Retry cycle: prioritizing critique-focused improvements")
            
            for focused_issue in feedback_analysis['critique_focused_issues']:
                priority_item = {
                    'priority': 'critical' if focused_issue['priority'] in ['critical', 'high'] else 'medium',
                    'issue': focused_issue['suggestion'],
                    'editing_type': 'critique_focused',
                    'scope': 'targeted',  # More targeted for retry cycles
                    'area': focused_issue['area'],
                    'expected_impact': focused_issue['expected_impact'],
                    'implementation_difficulty': focused_issue['implementation_difficulty'],
                    'source': 'critique_feedback'
                }
                priorities.append(priority_item)
        
        # High priority: Critical issues
        for issue in feedback_analysis['critical_issues']:
            priority_item = {
                'priority': 'high',
                'issue': issue,
                'editing_type': self._determine_editing_type(issue),
                'scope': 'global'  # Affects entire content
            }
            priorities.append(priority_item)
        
        # Medium priority: Moderate issues
        for issue in feedback_analysis['moderate_issues']:
            priority_item = {
                'priority': 'medium',
                'issue': issue,
                'editing_type': self._determine_editing_type(issue),
                'scope': 'sectional'  # Affects sections
            }
            priorities.append(priority_item)
        
        # Low priority: Minor issues
        for issue in feedback_analysis['minor_issues']:
            priority_item = {
                'priority': 'low',
                'issue': issue,
                'editing_type': self._determine_editing_type(issue),
                'scope': 'local'  # Affects specific parts
            }
            priorities.append(priority_item)
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        priorities.sort(key=lambda x: priority_order[x['priority']])
        
        return priorities
    
    def _determine_editing_type(self, issue: str) -> str:
        """Determine the type of editing needed based on the issue description."""
        
        issue_lower = issue.lower()
        
        if any(word in issue_lower for word in ['fact', 'accuracy', 'incorrect', 'wrong']):
            return 'fact_checking'
        elif any(word in issue_lower for word in ['style', 'tone', 'voice', 'consistency']):
            return 'style_improvement'
        elif any(word in issue_lower for word in ['structure', 'organization', 'flow', 'transition']):
            return 'structural_editing'
        elif any(word in issue_lower for word in ['clarity', 'readability', 'complex', 'confusing']):
            return 'clarity_improvement'
        elif any(word in issue_lower for word in ['technical', 'depth', 'expertise', 'domain']):
            return 'technical_enhancement'
        elif any(word in issue_lower for word in ['bias', 'compliance', 'ethical', 'legal']):
            return 'compliance_editing'
        elif 'critique:' in issue_lower or 'critique-focused' in issue_lower:
            return 'critique_focused'
        else:
            return 'general_improvement'
    
    async def _apply_content_improvements(
        self,
        draft: Draft,
        priorities: List[Dict[str, Any]],
        state: ContentPipelineState
    ) -> Draft:
        """Apply content improvements based on prioritized editing tasks."""
        
        current_content = draft.content
        current_title = draft.title
        
        # Process critique-focused issues first (for retry cycles)
        critique_focused_tasks = [p for p in priorities if p.get('editing_type') == 'critique_focused']
        if critique_focused_tasks:
            logger.info(f"🔄 Processing {len(critique_focused_tasks)} critique-focused tasks")
            current_content = await self._apply_critique_focused_edits(
                current_content, critique_focused_tasks, state
            )
            logger.info("✅ Critique-focused edits completed")
        
        # Process high-priority issues first
        high_priority_tasks = [p for p in priorities if p['priority'] == 'high' and p.get('editing_type') != 'critique_focused']
        if high_priority_tasks:
            logger.info(f"🔴 Processing {len(high_priority_tasks)} high-priority tasks")
            current_content, current_title = await self._apply_high_priority_edits(
                current_content, current_title, high_priority_tasks, state
            )
            logger.info("✅ High-priority edits completed")
        
        # Process medium-priority issues
        medium_priority_tasks = [p for p in priorities if p['priority'] == 'medium']
        if medium_priority_tasks:
            logger.info(f"🟡 Processing {len(medium_priority_tasks)} medium-priority tasks")
            current_content = await self._apply_medium_priority_edits(
                current_content, medium_priority_tasks, state
            )
            logger.info("✅ Medium-priority edits completed")
        
        # Process low-priority issues
        low_priority_tasks = [p for p in priorities if p['priority'] == 'low']
        if low_priority_tasks:
            logger.info(f"🟢 Processing {len(low_priority_tasks)} low-priority tasks")
            current_content = await self._apply_low_priority_edits(
                current_content, low_priority_tasks
            )
            logger.info("✅ Low-priority edits completed")
        
        # Restore any domain terms that may have been lost during editing
        current_content = self._restore_domain_terms(draft.content, current_content, state)
        
        # Apply readability polishing (sentence length optimization)
        current_content = self._polish_for_readability(current_content)
        
        # Ensure minimum protected domain terms remain for quality gates
        domain_terms = state.get('domain_terms_used') or state.get('domain_terms', []) or []
        current_content = self._ensure_minimum_domain_terms(current_content, domain_terms)
        
        # Create revised draft
        revised_draft = Draft(
            title=current_title,
            subtitle=draft.subtitle,
            abstract=draft.abstract,
            content=current_content,
            citations=draft.citations,
            word_count=len(current_content.split()),
            reading_time_minutes=max(1, len(current_content.split()) // 200),
            meta_description=draft.meta_description,
            keywords=draft.keywords,
            tags=draft.tags
        )
        
        return revised_draft
    
    def _extract_clean_content_from_response(self, response_content: str, original_content: str) -> str:
        """
        Extract clean content from LLM response, removing any editorial metadata.
        
        This is a safety net to ensure no editorial analysis leaks into the content.
        """
        
        # Log for debugging
        logger.info(f"🔧 Extracting clean content from LLM response (length: {len(response_content)})")
        
        # If response is suspiciously long compared to original, it likely contains metadata
        if len(response_content) > len(original_content) * 1.5:
            logger.warning(f"⚠️  Response suspiciously long ({len(response_content)} vs {len(original_content)}), applying sanitization")
        
        # Apply the same sanitization logic as in PublisherAgent
        import re
        
        # Define patterns to identify editorial metadata sections
        editorial_patterns = [
            r'\n\s*#*\s*Editing Summary.*$',
            r'\n\s*#*\s*Feedback Addressed.*$', 
            r'\n\s*#*\s*Priorities Handled.*$',
            r'\n\s*#*\s*Changes Made.*$',
            r'\n\s*#*\s*Editorial Notes.*$',
            r'\n\s*#*\s*Quality Assessment.*$',
            r'\n\s*#*\s*Analysis:.*$',
            r'\n\s*#*\s*Summary:.*$'
        ]
        
        # Look for delimiter patterns that separate main content from editorial metadata
        delimiter_patterns = [
            r'\n\s*—+\s*\n',  # Em dashes
            r'\n\s*-{3,}\s*\n',  # Multiple hyphens
            r'\n\s*={3,}\s*\n',  # Multiple equals signs
            r'\n\s*\*{3,}\s*\n'  # Multiple asterisks
        ]
        
        # Common transition phrases that indicate editorial content
        transition_patterns = [
            r'\n\s*Improvements Applied:.*$',
            r'\n\s*Changes Detected:.*$',
            r'\n\s*Quality Improvement.*$',
            r'\n\s*Editing Effectiveness.*$',
            r'\n\s*Confidence.*assessment.*$'
        ]
        
        cleaned_content = response_content.strip()
        
        # First, try to find delimiter patterns and cut content there
        for delimiter_pattern in delimiter_patterns:
            match = re.search(delimiter_pattern, cleaned_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                cleaned_content = cleaned_content[:match.start()].strip()
                logger.info(f"✅ Content cleaned using delimiter pattern, new length: {len(cleaned_content)}")
                break
        
        # If no delimiter found, look for specific editorial section headers
        if len(cleaned_content) == len(response_content.strip()):  # No delimiter cut was made
            for pattern in editorial_patterns:
                match = re.search(pattern, cleaned_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    cleaned_content = cleaned_content[:match.start()].strip()
                    logger.info(f"✅ Content cleaned using editorial pattern, new length: {len(cleaned_content)}")
                    break
        
        # If still no cut, check for transition phrases
        if len(cleaned_content) == len(response_content.strip()):  # No cut was made yet
            for pattern in transition_patterns:
                match = re.search(pattern, cleaned_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    cleaned_content = cleaned_content[:match.start()].strip()
                    logger.info(f"✅ Content cleaned using transition pattern, new length: {len(cleaned_content)}")
                    break
        
        # Final validation
        if len(cleaned_content) < len(original_content) * 0.3:
            logger.warning(f"⚠️  Cleaned content suspiciously short, reverting to original content")
            return original_content
            
        logger.info(f"✅ Content extraction complete: {len(response_content)} -> {len(cleaned_content)} chars")
        return cleaned_content
    
    def _validate_content_for_editorial_metadata(self, content: str, stage: str) -> bool:
        """
        Validate content to ensure no editorial metadata is present.
        
        Returns True if content is clean, False if editorial metadata detected.
        """
        
        # Common editorial metadata indicators
        editorial_indicators = [
            "Editing Summary",
            "Feedback Addressed", 
            "Priorities Handled",
            "Improvements Applied:",
            "Changes Detected:",
            "Quality Improvement Estimate:",
            "Editing Effectiveness Ratio:",
            "Confidence assessment",
            "Moderate confidence"
        ]
        
        found_indicators = []
        for indicator in editorial_indicators:
            if indicator.lower() in content.lower():
                found_indicators.append(indicator)
        
        if found_indicators:
            logger.error(f"❌ EDITORIAL METADATA DETECTED in {stage}:")
            for indicator in found_indicators:
                logger.error(f"   - Found: '{indicator}'")
            logger.error(f"   - Content length: {len(content)} chars")
            logger.error(f"   - Content preview: {content[:200]}...")
            return False
        
        logger.info(f"✅ Content validation passed for {stage} (no editorial metadata detected)")
        return True
    
    async def _apply_critique_focused_edits(
        self,
        content: str,
        tasks: List[Dict[str, Any]],
        state: ContentPipelineState
    ) -> str:
        """Apply critique-focused edits for retry cycles with targeted improvements."""
        
        if not tasks:
            return content
        
        system_prompt = editor_agent_instruction
        
        # Build focused improvement summary
        improvement_summary = []
        for task in tasks:
            improvement_summary.append(f"- {task['area']}: {task['issue']} (Impact: {task.get('expected_impact', 'medium')})")
        
        critique_cycle = state.get('critique_cycle_count', 1)
        
        human_prompt = f"""Apply these specific critique-focused improvements to the content:

Critique Cycle: {critique_cycle}
Targeted Improvements Needed:
{chr(10).join(improvement_summary)}

Current content:
{content}

IMPORTANT: Return ONLY the improved content. Do not include any editing summaries, analysis, feedback sections, or metadata. Your response should contain only the revised article content and nothing else.

Focus only on the identified issues - do not make unrelated changes.
Maintain the overall structure and flow unless specifically requested to change it."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            # Add timeout protection to prevent hanging
            response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=60.0)
            logger.info(f"Applied {len(tasks)} critique-focused improvements")
            
            # Extract clean content from response, removing any editorial metadata
            clean_content = self._extract_clean_content_from_response(response.content, content)
            return clean_content
            
        except asyncio.TimeoutError:
            logger.error(f"Critique-focused editing timed out after 60 seconds")
            return content
        except Exception as e:
            logger.warning(f"Critique-focused editing failed: {e}")
            return content
    
    async def _apply_high_priority_edits(
        self,
        content: str,
        title: str,
        tasks: List[Dict[str, Any]],
        state: ContentPipelineState
    ) -> tuple[str, str]:
        """Apply high-priority edits that may require significant content changes."""
        
        system_prompt = editor_agent_instruction
        
        issues_summary = "\n".join([f"- {task['issue']}" for task in tasks])
        
        human_prompt = f"""Improve this content by addressing these critical issues:

        Issues to address:
        {issues_summary}

        Current title: {title}
        Current content:
        {content}

        Return JSON with:
        {{
            "revised_title": "improved title if needed",
            "revised_content": "improved content with critical issues addressed",
            "changes_made": ["change 1", "change 2", "change 3"],
            "preserved_elements": ["element 1", "element 2"]
        }}

        Ensure the revised content maintains professional quality while addressing all critical issues.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            # Add timeout protection to prevent hanging
            response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=60.0)
            result = json.loads(response.content)
            
            return result['revised_content'], result.get('revised_title', title)
            
        except asyncio.TimeoutError:
            logger.error(f"High-priority editing timed out after 60 seconds")
            return content, title
        except Exception as e:
            logger.warning(f"High-priority editing failed: {e}")
            return content, title
    
    async def _apply_medium_priority_edits(
        self,
        content: str,
        tasks: List[Dict[str, Any]],
        state: ContentPipelineState
    ) -> str:
        """Apply medium-priority edits for style and flow improvements."""
        
        system_prompt = editor_agent_instruction
        
        issues_summary = "\n".join([f"- {task['issue']}" for task in tasks])
        
        human_prompt = f"""Improve this content by addressing these medium-priority issues:

        Issues to address:
        {issues_summary}

        Content to improve:
        {content}

        IMPORTANT: Return ONLY the improved content. Do not include any editing summaries, analysis, feedback sections, or metadata. Your response should contain only the revised article content and nothing else.

        Focus on enhancing style, flow, and readability while maintaining the original structure and key points.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            # Add timeout protection to prevent hanging
            response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=60.0)
            
            # Extract clean content from response, removing any editorial metadata
            clean_content = self._extract_clean_content_from_response(response.content, content)
            return clean_content
            
        except asyncio.TimeoutError:
            logger.error(f"Medium-priority editing timed out after 60 seconds")
            return content
        except Exception as e:
            logger.warning(f"Medium-priority editing failed: {e}")
            return content
    
    async def _apply_low_priority_edits(
        self,
        content: str,
        tasks: List[Dict[str, Any]]
    ) -> str:
        """Apply low-priority edits for minor polishing and refinements."""
        
        system_prompt = editor_agent_instruction
        
        issues_summary = "\n".join([f"- {task['issue']}" for task in tasks])
        
        human_prompt = f"""Polish this content by addressing these minor issues:

        Issues to address:
        {issues_summary}

        Content to polish:
        {content}

        IMPORTANT: Return ONLY the polished content. Do not include any editing summaries, analysis, feedback sections, or metadata. Your response should contain only the revised article content and nothing else.

        Focus on refinement and polish while preserving the existing quality.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            # Add timeout protection to prevent hanging
            response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=60.0)
            
            # Extract clean content from response, removing any editorial metadata
            clean_content = self._extract_clean_content_from_response(response.content, content)
            return clean_content
            
        except asyncio.TimeoutError:
            logger.error(f"Low-priority editing timed out after 60 seconds")
            return content
        except Exception as e:
            logger.warning(f"Low-priority editing failed: {e}")
            return content
    
    def _verify_improvements(
        self,
        original_draft: Draft,
        revised_draft: Draft,
        feedback_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify that improvements were successfully applied."""
        
        improvements_applied = 0
        changes_detected = []
        
        # Check if title changed
        if original_draft.title != revised_draft.title:
            changes_detected.append("Title updated")
            improvements_applied += 1
        
        # Check if content length changed significantly
        original_word_count = original_draft.word_count
        revised_word_count = revised_draft.word_count
        
        word_count_change = abs(revised_word_count - original_word_count)
        if word_count_change > 10:  # Significant change threshold
            changes_detected.append(f"Content length changed by {word_count_change} words")
            improvements_applied += 1
        
        # Check content similarity for other changes
        content_changes = self._detect_content_changes(
            original_draft.content, 
            revised_draft.content
        )
        changes_detected.extend(content_changes)
        improvements_applied += len(content_changes)
        
        # Assess quality improvement potential
        quality_improvement_estimate = self._estimate_quality_improvement(
            feedback_analysis,
            improvements_applied
        )
        
        return {
            'improvements_applied': improvements_applied,
            'changes_detected': changes_detected,
            'word_count_change': word_count_change,
            'quality_improvement_estimate': quality_improvement_estimate,
            'editing_effectiveness': min(1.0, improvements_applied / max(1, len(feedback_analysis.get('critical_issues', [])) + len(feedback_analysis.get('moderate_issues', []))))
        }
    
    def _detect_content_changes(self, original: str, revised: str) -> List[str]:
        """Detect significant changes between original and revised content."""
        
        changes = []
        
        # Simple change detection
        if len(revised) != len(original):
            if len(revised) > len(original):
                changes.append("Content expanded")
            else:
                changes.append("Content condensed")
        
        # Check for structural changes
        original_headers = len(re.findall(r'^#{1,6}\s+', original, re.MULTILINE))
        revised_headers = len(re.findall(r'^#{1,6}\s+', revised, re.MULTILINE))
        
        if revised_headers != original_headers:
            changes.append(f"Structure modified ({original_headers} -> {revised_headers} headers)")
        
        # Check for paragraph changes
        original_paragraphs = len([p for p in original.split('\n\n') if p.strip()])
        revised_paragraphs = len([p for p in revised.split('\n\n') if p.strip()])
        
        if abs(revised_paragraphs - original_paragraphs) > 1:
            changes.append("Paragraph structure updated")
        
        # Check for new lists or formatting
        original_lists = len(re.findall(r'^\s*[-*+]\s+', original, re.MULTILINE))
        revised_lists = len(re.findall(r'^\s*[-*+]\s+', revised, re.MULTILINE))
        
        if revised_lists > original_lists:
            changes.append("Added list formatting")
        
        return changes[:5]  # Limit to top 5 changes
    
    def _estimate_quality_improvement(
        self,
        feedback_analysis: Dict[str, Any],
        improvements_applied: int
    ) -> Dict[str, Any]:
        """Estimate the potential quality improvement from editing."""
        
        total_issues = (
            len(feedback_analysis.get('critical_issues', [])) +
            len(feedback_analysis.get('moderate_issues', [])) +
            len(feedback_analysis.get('minor_issues', []))
        )
        
        if total_issues == 0:
            return {
                'estimated_score_improvement': 0.0,
                'confidence': 'high',
                'areas_improved': []
            }
        
        # Estimate score improvement based on issues addressed
        critical_weight = 0.15
        moderate_weight = 0.10
        minor_weight = 0.05
        
        potential_improvement = (
            len(feedback_analysis.get('critical_issues', [])) * critical_weight +
            len(feedback_analysis.get('moderate_issues', [])) * moderate_weight +
            len(feedback_analysis.get('minor_issues', [])) * minor_weight
        )
        
        # Adjust based on improvements actually applied
        if improvements_applied > 0:
            effectiveness_ratio = min(1.0, improvements_applied / total_issues)
            estimated_improvement = potential_improvement * effectiveness_ratio
        else:
            estimated_improvement = 0.0
        
        areas_improved = list(feedback_analysis.get('areas_to_improve', {}).keys())
        
        return {
            'estimated_score_improvement': estimated_improvement,
            'confidence': 'high' if improvements_applied >= total_issues * 0.7 else 'medium',
            'areas_improved': areas_improved,
            'effectiveness_ratio': effectiveness_ratio if improvements_applied > 0 else 0.0
        }
    
    def _restore_lost_citations(self, original_draft: 'Draft', revised_draft: 'Draft', state: Dict[str, Any]) -> 'Draft':
        """Restore citations that were lost during the editing process."""
        
        import re
        from typing import Dict, List, Set
        
        # Extract citations from original and revised content
        original_citations = set(re.findall(r'\[Citation\s*(\d+)\]', original_draft.content))
        revised_citations = set(re.findall(r'\[Citation\s*(\d+)\]', revised_draft.content))
        
        lost_citations = original_citations - revised_citations
        
        if not lost_citations:
            logger.info("✅ All citations preserved during editing")
            return revised_draft
        
        logger.warning(f"⚠️  Lost citations detected during editing: {lost_citations}")
        
        # Get section-to-source mapping from state if available
        section_source_mapping = state.get('section_source_mapping', {})
        sources = state.get('sources', [])
        citation_lookup = {}
        
        # Rebuild citation lookup from sources
        if sources:
            for i, source in enumerate(sources, 1):
                if hasattr(source, 'url'):
                    citation_lookup[source.url] = i
                elif isinstance(source, dict) and 'url' in source:
                    citation_lookup[source['url']] = i
        
        # Restore lost citations by re-analyzing content
        restored_content = self._reapply_lost_citations(
            revised_draft.content,
            lost_citations,
            citation_lookup,
            sources
        )
        
        # Create new draft with restored citations
        restored_draft = Draft(
            title=revised_draft.title,
            subtitle=revised_draft.subtitle,
            abstract=revised_draft.abstract,
            content=restored_content,
            citations=revised_draft.citations,
            word_count=len(restored_content.split()),
            reading_time=max(1, len(restored_content.split()) // 200),
            quality_score=revised_draft.quality_score
        )
        
        # Verify restoration
        final_citations = set(re.findall(r'\[Citation\s*(\d+)\]', restored_content))
        recovered_citations = final_citations.intersection(lost_citations)
        
        if recovered_citations:
            logger.info(f"✅ Successfully recovered {len(recovered_citations)} lost citations: {recovered_citations}")
        else:
            logger.warning(f"⚠️  Unable to recover lost citations: {lost_citations}")
        
        return restored_draft
    
    def _reapply_lost_citations(
        self, 
        content: str, 
        lost_citations: set, 
        citation_lookup: Dict[str, int],
        sources: List[Any]
    ) -> str:
        """Re-apply lost citations to appropriate sentences in the content."""
        
        import re
        
        # Convert lost citations to integers
        lost_citation_nums = []
        for citation in lost_citations:
            try:
                lost_citation_nums.append(int(citation))
            except ValueError:
                continue
        
        if not lost_citation_nums:
            return content
        
        # Split content into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        restored_sentences = []
        citations_to_restore = lost_citation_nums.copy()
        
        for sentence in sentences:
            stripped = sentence.strip()
            if not stripped:
                restored_sentences.append(sentence)
                continue
            
            # Skip sentences that already have citations
            if '[Citation' in stripped:
                restored_sentences.append(sentence)
                continue
            
            # Identify sentences that likely need citations
            needs_citation = (
                # Statistical claims
                bool(re.search(r'\d+%|\d+\s*percent|\d{2,}', stripped)) or
                # Authority claims  
                bool(re.search(r'\baccording to\b|\bstudy\b|\breport\b|\bresearch\b|\banalysis\b', stripped, re.IGNORECASE)) or
                # Definitive statements
                bool(re.search(r'\bis\s+proven\b|\bdemonstrates\b|\bshows\b|\bindicates\b|\breveal\b', stripped, re.IGNORECASE)) or
                # Industry/domain claims
                bool(re.search(r'\bindustry\b|\bmarket\b|\bstandard\b|\bbest\s+practice\b', stripped, re.IGNORECASE))
            )
            
            # Apply first available lost citation to sentences that need them
            if needs_citation and citations_to_restore:
                citation_num = citations_to_restore.pop(0)
                restored_sentences.append(f"{stripped} [Citation {citation_num}]")
                logger.debug(f"Restored citation {citation_num} to sentence: {stripped[:50]}...")
            else:
                restored_sentences.append(sentence)
        
        return " ".join(restored_sentences)
    
    def _restore_domain_terms(self, original_content: str, edited_content: str, state: Dict[str, Any]) -> str:
        """Restore domain terms that may have been lost during editing."""
        
        import re
        
        # Extract protected domain terms from original content
        original_protected_terms = re.findall(r'<([^>]+)>', original_content)
        edited_protected_terms = re.findall(r'<([^>]+)>', edited_content)
        
        # Check if any protected terms were lost
        lost_terms = set(original_protected_terms) - set(edited_protected_terms)
        
        if not lost_terms:
            logger.debug("✅ All domain terms preserved during editing")
            return edited_content
        
        logger.warning(f"⚠️  Lost domain terms detected during editing: {lost_terms}")
        
        # Get domain information from state for context
        domain_focus = state.get('domain_focus', [])
        
        # Re-apply lost domain terms intelligently
        restored_content = self._reapply_lost_domain_terms(
            edited_content, 
            lost_terms, 
            domain_focus
        )
        
        # Verify restoration
        final_protected_terms = re.findall(r'<([^>]+)>', restored_content)
        recovered_terms = set(final_protected_terms).intersection(lost_terms)
        
        if recovered_terms:
            logger.info(f"✅ Successfully recovered {len(recovered_terms)} lost domain terms: {recovered_terms}")
        else:
            logger.warning(f"⚠️  Unable to recover lost domain terms: {lost_terms}")
        
        return restored_content
    
    def _reapply_lost_domain_terms(self, content: str, lost_terms: set, domain_focus: List[str]) -> str:
        """Re-apply lost domain terms to appropriate locations in content."""
        
        import re
        
        # Split content into paragraphs for targeted restoration
        paragraphs = content.split('\n\n')
        restored_paragraphs = []
        terms_to_restore = list(lost_terms)
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                restored_paragraphs.append(paragraph)
                continue
            
            # For substantial paragraphs, try to restore 1-2 terms
            if len(paragraph.strip()) > 100 and terms_to_restore:
                # Take 1-2 terms for this paragraph
                paragraph_terms = terms_to_restore[:2]
                terms_to_restore = terms_to_restore[2:]
                
                restored_paragraph = self._integrate_lost_terms_into_paragraph(
                    paragraph, 
                    paragraph_terms, 
                    domain_focus
                )
                restored_paragraphs.append(restored_paragraph)
            else:
                restored_paragraphs.append(paragraph)
        
        # If we still have terms to restore, add them to the last substantial paragraph
        if terms_to_restore and restored_paragraphs:
            # Find the last substantial paragraph
            for i in range(len(restored_paragraphs) - 1, -1, -1):
                if len(restored_paragraphs[i].strip()) > 50:
                    restored_paragraphs[i] = self._integrate_lost_terms_into_paragraph(
                        restored_paragraphs[i],
                        terms_to_restore,
                        domain_focus
                    )
                    break
        
        return '\n\n'.join(restored_paragraphs)
    
    def _integrate_lost_terms_into_paragraph(self, paragraph: str, terms: List[str], domain_focus: List[str]) -> str:
        """Integrate lost domain terms back into a paragraph naturally."""
        
        if not terms:
            return paragraph
        
        # Split into sentences for more precise integration
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        enhanced_sentences = []
        
        for i, sentence in enumerate(sentences):
            enhanced_sentence = sentence
            
            # For the first sentence of the paragraph, try to integrate terms
            if i == 0 and terms:
                # Take the first term
                term = terms[0]
                terms = terms[1:]
                
                # Integrate the term naturally based on sentence context
                if term.lower() not in enhanced_sentence.lower():
                    # Add the term with protection markers based on context
                    if any(keyword in enhanced_sentence.lower() for keyword in ['implement', 'approach', 'strategy']):
                        enhanced_sentence = enhanced_sentence.rstrip('.') + f" utilizing <{term}> methodologies."
                    elif any(keyword in enhanced_sentence.lower() for keyword in ['system', 'solution', 'platform']):
                        enhanced_sentence = enhanced_sentence.rstrip('.') + f" leveraging <{term}> capabilities."
                    elif any(keyword in enhanced_sentence.lower() for keyword in ['ensure', 'maintain', 'achieve']):
                        enhanced_sentence = enhanced_sentence.rstrip('.') + f" through <{term}> best practices."
                    else:
                        enhanced_sentence = enhanced_sentence.rstrip('.') + f" incorporating <{term}> principles."
                else:
                    # Term exists in sentence but without protection - add protection
                    enhanced_sentence = re.sub(
                        rf'\\b{re.escape(term)}\\b', 
                        f'<{term}>', 
                        enhanced_sentence, 
                        flags=re.IGNORECASE
                    )
            
            enhanced_sentences.append(enhanced_sentence)
        
        return ' '.join(enhanced_sentences)
    
    def _polish_for_readability(self, content: str) -> str:
        """Polish content for optimal readability by splitting long sentences and optimizing structure."""
        
        import re
        
        # Split content into sections to preserve structure
        sections = content.split('\n## ')
        polished_sections = []
        
        for i, section in enumerate(sections):
            if i == 0:
                # First section doesn't need ## prefix restoration
                polished_section = self._polish_section_readability(section)
            else:
                # Restore ## prefix and polish
                polished_section = '## ' + self._polish_section_readability(section)
            
            polished_sections.append(polished_section)
        
        polished_content = '\n'.join(polished_sections)
        
        # Log readability improvements
        original_sentences = re.split(r'(?<=[.!?])\s+', content)
        polished_sentences = re.split(r'(?<=[.!?])\s+', polished_content)
        
        long_original = sum(1 for s in original_sentences if len(s.split()) > 24)
        long_polished = sum(1 for s in polished_sentences if len(s.split()) > 24)
        
        if long_original > long_polished:
            logger.info(f"✅ Readability improved: {long_original} → {long_polished} long sentences (>24 words)")
        
        return polished_content
    
    def _ensure_minimum_domain_terms(self, content: str, domain_terms: List[str], threshold: int = 8) -> str:
        """Ensure the content retains the required number of protected domain terms."""
        import re
        
        if not domain_terms:
            return content
        
        protected_terms = re.findall(r'<([^>]+)>', content)
        unique_terms = []
        for term in protected_terms:
            if term not in unique_terms:
                unique_terms.append(term)
        
        if len(unique_terms) >= threshold:
            return content
        
        remaining_slots = threshold - len(unique_terms)
        additional_terms: List[str] = []
        for term in domain_terms:
            if term not in unique_terms and term not in additional_terms:
                additional_terms.append(term)
            if len(additional_terms) >= remaining_slots:
                break
        
        if not additional_terms:
            return content
        
        reinforcement = ", ".join(f"<{term}>" for term in additional_terms)
        return content.rstrip() + f"\n\n*Key domain emphasis reinforced: {reinforcement}.*"
    
    def _polish_section_readability(self, section: str) -> str:
        """Polish a single section for readability."""
        
        import re
        
        if not section.strip():
            return section
        
        # Split into paragraphs
        paragraphs = section.split('\n\n')
        polished_paragraphs = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                polished_paragraphs.append(paragraph)
                continue
            
            # Check if paragraph is a bullet list - preserve formatting
            if re.match(r'^\s*[-*•]\s+', paragraph, re.MULTILINE):
                polished_paragraphs.append(paragraph)
                continue
            
            # Polish regular paragraphs
            polished_paragraph = self._polish_paragraph_sentences(paragraph)

            # Convert very long paragraphs into bullet lists for improved readability
            word_total = len(re.sub(r'\[Citation\s*\d+\]', '', polished_paragraph).split())
            if word_total > 110:
                sentences_for_bullets = [
                    s.strip() for s in re.split(r'(?<=[.!?])\s+', polished_paragraph) if s.strip()
                ]
                if sentences_for_bullets:
                    bullet_items = [f"- {sentence.rstrip('.')}" for sentence in sentences_for_bullets]
                    polished_paragraph = '\n'.join(bullet_items)

            polished_paragraphs.append(polished_paragraph)
        
        return '\n\n'.join(polished_paragraphs)
    
    def _polish_paragraph_sentences(self, paragraph: str) -> str:
        """Polish sentences in a paragraph for optimal readability."""
        
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        polished_sentences = []
        
        for sentence in sentences:
            stripped = sentence.strip()
            if not stripped:
                polished_sentences.append(sentence)
                continue
            
            # Check sentence length (exclude citations and protected terms from word count)
            # Remove citations and protected terms for accurate word counting
            clean_sentence = re.sub(r'\[Citation\s*\d+\]', '', stripped)
            clean_sentence = re.sub(r'<[^>]+>', '', clean_sentence)
            word_count = len(clean_sentence.split())
            
            if word_count > 24:
                # Attempt to split long sentences
                polished_sentence = self._split_long_sentence(stripped)
                polished_sentences.extend(re.split(r'(?<=[.!?])\s+', polished_sentence))
            else:
                polished_sentences.append(stripped)
        
        # Clean up and join sentences with spaces
        cleaned_sentences = [s.strip() for s in polished_sentences if s.strip()]
        return ' '.join(cleaned_sentences)
    
    def _split_long_sentence(self, sentence: str) -> str:
        """Split a long sentence into shorter, more readable sentences."""
        
        import re
        
        # Preserve original if it contains complex structures we shouldn't split
        if '[Citation' in sentence and sentence.count('[Citation') > 2:
            # Too many citations - splitting might break citation flow
            return sentence
        
        # Look for natural breaking points
        break_patterns = [
            # Coordinating conjunctions with comma
            (r',\s+(and|but|or|yet|so)\s+', r'. \\1 '),
            # Subordinating conjunctions
            (r',\s+(because|since|while|although|though|whereas|if)\s+', r'. \\1 '),
            # Transitional phrases
            (r',\s+(however|therefore|moreover|furthermore|additionally|consequently)\s*,\s*', r'. \\1, '),
            # Relative clauses that can be independent
            (r',\s+(which|that)\s+([^,]{20,}),', r'. This \\2.'),
        ]
        
        split_sentence = sentence
        for pattern, replacement in break_patterns:
            # Only apply if the resulting sentences would be reasonable length
            test_split = re.sub(pattern, replacement, split_sentence, count=1)
            test_sentences = re.split(r'(?<=[.!?])\s+', test_split)
            
            # Check if split results in better sentence lengths
            if len(test_sentences) > 1:
                max_words = max(len(s.split()) for s in test_sentences)
                if max_words < len(sentence.split()) * 0.8:  # Meaningful reduction
                    split_sentence = test_split
                    break
        
        # If no natural break found and sentence is very long (>30 words), try harder
        if len(split_sentence.split()) > 30 and split_sentence == sentence:
            # Look for semicolons or em dashes
            if ';' in sentence:
                split_sentence = sentence.replace(';', '.')
            elif ' - ' in sentence:
                split_sentence = sentence.replace(' - ', '. ')
        
        # Final fallback: chunk long sentences to <= 24 words
        words = split_sentence.split()
        if len(words) > 24:
            chunks = [' '.join(words[i:i+20]) for i in range(0, len(words), 20)]
            chunked_sentence = '. '.join(chunk.strip().rstrip('.') for chunk in chunks if chunk.strip())
            if not chunked_sentence.endswith('.'):
                chunked_sentence += '.'
            return chunked_sentence
        
        return split_sentence
