"""Content Editor Agent - Improves content based on quality feedback and guidelines."""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
import re
import json

from ..state.pipeline_state import Draft, QualityScores, ContentPipelineState

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
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.4,  # Balanced creativity for editing
            max_tokens=4000
        )
    
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
        logger.info("Starting content editing based on quality feedback")
        
        try:
            # Step 1: Analyze quality feedback (including critique feedback)
            feedback_analysis = self._analyze_quality_feedback(quality_feedback, state)
            
            # Step 2: Prioritize editing tasks based on critique feedback
            editing_priorities = self._prioritize_editing_tasks(feedback_analysis, state)
            
            # Step 3: Apply content improvements
            revised_draft = await self._apply_content_improvements(
                draft, 
                editing_priorities,
                state
            )
            
            # Step 4: Verify improvements
            improvement_summary = self._verify_improvements(draft, revised_draft, feedback_analysis)
            
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
            current_content = await self._apply_critique_focused_edits(
                current_content, critique_focused_tasks, state
            )
        
        # Process high-priority issues first
        high_priority_tasks = [p for p in priorities if p['priority'] == 'high' and p.get('editing_type') != 'critique_focused']
        if high_priority_tasks:
            current_content, current_title = await self._apply_high_priority_edits(
                current_content, current_title, high_priority_tasks, state
            )
        
        # Process medium-priority issues
        medium_priority_tasks = [p for p in priorities if p['priority'] == 'medium']
        if medium_priority_tasks:
            current_content = await self._apply_medium_priority_edits(
                current_content, medium_priority_tasks, state
            )
        
        # Process low-priority issues
        low_priority_tasks = [p for p in priorities if p['priority'] == 'low']
        if low_priority_tasks:
            current_content = await self._apply_low_priority_edits(
                current_content, low_priority_tasks
            )
        
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

Current content: {content[:2500]}{"..." if len(content) > 2500 else ""}

Return the improved content with the specific critique points addressed.
Focus only on the identified issues - do not make unrelated changes.
Maintain the overall structure and flow unless specifically requested to change it."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            logger.info(f"Applied {len(tasks)} critique-focused improvements")
            return response.content
            
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
        Current content: {content[:3000]}{"..." if len(content) > 3000 else ""}

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
            response = await self.llm.ainvoke(messages)
            result = json.loads(response.content)
            
            return result['revised_content'], result.get('revised_title', title)
            
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
        {content[:2500]}{"..." if len(content) > 2500 else ""}

        Return the improved content that addresses these issues while maintaining the original structure and key points.
        Focus on enhancing style, flow, and readability.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
            
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
        {content[:2000]}{"..." if len(content) > 2000 else ""}

        Return the polished content with minor improvements applied.
        Focus on refinement and polish while preserving the existing quality.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
            
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