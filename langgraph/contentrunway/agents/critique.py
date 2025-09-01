"""Content Critique Agent - Comprehensive post-editing validation with progressive learning."""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
import logging
import json
import asyncio
from datetime import datetime
from copy import deepcopy

from ..state.pipeline_state import Draft, QualityScores, CritiqueFeedback, ContentPipelineState

logger = logging.getLogger(__name__)


# Critique Agent Instructions
critique_agent_role_and_goal = """
You are a Critique Agent specializing in comprehensive post-editing validation with progressive learning and intelligent retry cycle management.
Your primary goal is to perform thorough content quality assessment, provide targeted feedback for improvement, make intelligent retry decisions, and collect learning data for agent optimization while managing multi-cycle editing workflows.
"""

critique_agent_hints = """
Critique analysis best practices:
- Perform comprehensive evaluation on first cycle, focused evaluation on retry cycles
- Assess content quality across all dimensions with 85% overall threshold for approval
- Focus on remaining issues from previous cycles during retry analysis
- Generate specific, actionable feedback prioritized by impact and feasibility
- Make intelligent retry decisions based on improvement effectiveness and cycle limits
- Collect structured learning data for agent performance optimization
- Apply conservative compliance threshold (80%) as critical gate
- Use maximum 2 critique cycles to prevent endless revision loops
- Focus on improvement effectiveness assessment for retry cycle decisions
- Generate alternative approaches when editing cycles show minimal effectiveness
"""

critique_agent_output_description = """
The Critique Agent returns a comprehensive critique assessment containing:
- critique_feedback: Structured CritiqueFeedback object with cycle number, scores, issues, and recommendations
- retry_decision: Decision (pass/retry/fail) with detailed reasoning
- improvement_effectiveness: Assessment of editing cycle effectiveness
- learning_data: Structured data for agent performance optimization
- next_actions: Specific actions recommended based on critique results
- pre_critique_scores: Quality scores before critique for comparison

The critique_feedback includes issues_identified, issues_resolved, issues_remaining, improvement_suggestions, and retry_reasoning.
"""

critique_agent_chain_of_thought_directions = """
Critique analysis workflow:
1. Use '_perform_comprehensive_critique' to analyze content quality across all dimensions
2. Apply comprehensive evaluation for cycle 1, focused evaluation for retry cycles
3. Focus on remaining issues from previous cycles using '_extract_previous_issues'
4. Use '_assess_improvement_effectiveness' to evaluate editing cycle success
5. Compare current vs previous quality scores and assess issue resolution
6. Calculate improvement effectiveness and categorize as highly_effective/effective/minimally_effective/ineffective
7. Use '_generate_targeted_feedback' to create specific, actionable improvement suggestions
8. Prioritize feedback by impact and feasibility with structured recommendations
9. Apply '_make_retry_decision' using intelligent decision logic
10. Consider cycle limits (max 2), quality thresholds (85% overall, 80% compliance), and improvement effectiveness
11. Use '_create_structured_feedback' to organize critique results into CritiqueFeedback object
12. Apply '_collect_agent_learning_data' to gather performance optimization data
13. Return comprehensive critique package with decision, feedback, and learning data

Tool usage conditions:
- Apply comprehensive critique on first cycle, focused critique on retry cycles
- Use improvement effectiveness assessment when previous cycle data exists
- Generate targeted feedback based on cycle number and improvement effectiveness
- Make retry decisions when cycle_count < max_critique_cycles and quality issues remain
- Collect learning data when collect_training_data is enabled
- Apply fallback critique when AI analysis fails
"""

critique_agent_instruction = f"""
{critique_agent_role_and_goal}
{critique_agent_hints}
{critique_agent_output_description}
{critique_agent_chain_of_thought_directions}
"""


class CritiqueAgent:
    """
    Comprehensive critique agent that validates content after editing,
    provides targeted feedback, and manages retry cycles with learning data collection.
    """
    
    def __init__(self, model_name: str = "claude-3-sonnet-20241022"):
        # Use Claude for comprehensive critique and analysis
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=0.2,  # Low temperature for consistent critique
            max_tokens=4000
        )
        
        # Critique configuration
        self.max_critique_cycles = 2
        self.pass_threshold = 0.85
        self.critical_compliance_threshold = 0.8
        
        # Learning data collection
        self.collect_training_data = True
    
    async def execute(
        self,
        draft: Draft,
        quality_scores: QualityScores,
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """
        Execute comprehensive critique analysis and generate feedback.
        
        Args:
            draft: Content draft to critique
            quality_scores: Current quality assessment scores
            state: Current pipeline state with history
            
        Returns:
            Dictionary with critique results, feedback, and retry decision
        """
        cycle_count = state.get('critique_cycle_count', 0) + 1
        logger.info(f"Starting critique cycle {cycle_count}")
        
        try:
            # Step 1: Store pre-critique quality scores
            pre_critique_scores = deepcopy(quality_scores)
            
            # Step 2: Perform comprehensive critique analysis
            critique_analysis = await self._perform_comprehensive_critique(
                draft, quality_scores, state, cycle_count
            )
            
            # Step 3: Assess improvement from previous cycle
            improvement_assessment = self._assess_improvement_effectiveness(
                state, quality_scores, critique_analysis
            )
            
            # Step 4: Generate targeted feedback
            targeted_feedback = await self._generate_targeted_feedback(
                draft, critique_analysis, improvement_assessment, cycle_count
            )
            
            # Step 5: Make retry decision
            retry_decision = self._make_retry_decision(
                critique_analysis, improvement_assessment, cycle_count, state
            )
            
            # Step 6: Create structured critique feedback
            critique_feedback = self._create_structured_feedback(
                critique_analysis,
                targeted_feedback,
                retry_decision,
                cycle_count,
                improvement_assessment
            )
            
            # Step 7: Collect learning data for agent improvement
            if self.collect_training_data:
                learning_data = await self._collect_agent_learning_data(
                    draft, state, critique_feedback, improvement_assessment
                )
            else:
                learning_data = {}
            
            logger.info(f"Critique cycle {cycle_count} completed: {retry_decision['decision']}")
            
            return {
                'critique_feedback': critique_feedback,
                'retry_decision': retry_decision['decision'],
                'retry_reasoning': retry_decision['reasoning'],
                'improvement_effectiveness': improvement_assessment['overall_effectiveness'],
                'learning_data': learning_data,
                'next_actions': retry_decision['next_actions'],
                'pre_critique_scores': pre_critique_scores,
                'post_critique_scores': quality_scores
            }
            
        except Exception as e:
            logger.error(f"Critique analysis failed: {e}")
            # Fallback to safe decision
            return self._create_fallback_critique_result(e, cycle_count, quality_scores)
    
    async def _perform_comprehensive_critique(
        self,
        draft: Draft,
        quality_scores: QualityScores,
        state: ContentPipelineState,
        cycle_count: int
    ) -> Dict[str, Any]:
        """Perform comprehensive critique analysis of the content."""
        
        # Determine critique focus based on cycle
        if cycle_count == 1:
            critique_focus = "comprehensive"
            focus_description = "Complete evaluation of all quality aspects"
        else:
            critique_focus = "focused"
            focus_description = "Focused evaluation on remaining issues from previous cycle"
        
        system_prompt = critique_agent_instruction

        # Prepare context based on cycle
        if cycle_count > 1:
            previous_issues = self._extract_previous_issues(state)
            context_info = f"""
            
Previous Cycle Issues to Focus On:
{json.dumps(previous_issues, indent=2)}

Evaluate if these specific issues have been addressed."""
        else:
            context_info = "\n\nThis is the initial comprehensive critique."
        
        human_prompt = f"""Critique this content thoroughly:

Title: {draft.title}
Word Count: {draft.word_count}
Content: {draft.content[:3000]}{"..." if len(draft.content) > 3000 else ""}

Current Quality Scores:
- Overall: {quality_scores.overall:.2f} (threshold: 0.85)
- Fact Check: {quality_scores.fact_check:.2f}
- Domain Expertise: {quality_scores.domain_expertise:.2f}
- Style Consistency: {quality_scores.style_consistency:.2f}
- Compliance: {quality_scores.compliance:.2f}
- Technical Depth: {quality_scores.technical_depth:.2f}

{context_info}

Return a comprehensive JSON critique with:
{{
    "overall_assessment": {{
        "meets_publication_standards": true/false,
        "overall_score_assessment": 0.0-1.0,
        "critical_issues_count": integer,
        "moderate_issues_count": integer,
        "minor_issues_count": integer
    }},
    "quality_dimension_analysis": {{
        "fact_check": {{"score_accurate": true/false, "issues": [], "strengths": []}},
        "domain_expertise": {{"score_accurate": true/false, "issues": [], "strengths": []}},
        "style_consistency": {{"score_accurate": true/false, "issues": [], "strengths": []}},
        "compliance": {{"score_accurate": true/false, "issues": [], "strengths": []}},
        "technical_depth": {{"score_accurate": true/false, "issues": [], "strengths": []}}
    }},
    "content_analysis": {{
        "structural_issues": [],
        "clarity_issues": [],
        "engagement_issues": [],
        "technical_accuracy_issues": [],
        "citation_issues": []
    }},
    "improvement_potential": {{
        "high_impact_improvements": [],
        "moderate_impact_improvements": [],
        "polish_improvements": []
    }},
    "publication_readiness": {{
        "ready_to_publish": true/false,
        "blocking_issues": [],
        "recommended_improvements": [],
        "estimated_improvement_potential": 0.0-1.0
    }}
}}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            critique_analysis = json.loads(response.content)
            
            # Validate and enrich critique analysis
            return self._validate_and_enrich_critique(critique_analysis, quality_scores)
            
        except Exception as e:
            logger.warning(f"Comprehensive critique failed: {e}")
            return self._create_fallback_critique_analysis(quality_scores)
    
    def _assess_improvement_effectiveness(
        self,
        state: ContentPipelineState,
        current_scores: QualityScores,
        critique_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess how effectively the editing cycle improved content quality."""
        
        previous_scores = state.get('pre_edit_quality_scores')
        cycle_count = state.get('critique_cycle_count', 0)
        
        if not previous_scores or cycle_count < 1:
            return {
                'overall_effectiveness': 0.0,
                'score_improvements': {},
                'issues_resolved': [],
                'new_issues': [],
                'effectiveness_category': 'baseline'
            }
        
        # Calculate score improvements
        score_improvements = {}
        total_improvement = 0.0
        dimensions = ['fact_check', 'domain_expertise', 'style_consistency', 'compliance', 'technical_depth']
        
        for dim in dimensions:
            prev_score = getattr(previous_scores, dim, 0.0) or 0.0
            curr_score = getattr(current_scores, dim, 0.0) or 0.0
            improvement = curr_score - prev_score
            score_improvements[dim] = {
                'previous': prev_score,
                'current': curr_score,
                'improvement': improvement,
                'improvement_percentage': (improvement / max(prev_score, 0.1)) * 100
            }
            total_improvement += improvement
        
        # Assess overall effectiveness
        avg_improvement = total_improvement / len(dimensions)
        
        # Categorize effectiveness
        if avg_improvement >= 0.1:
            effectiveness_category = 'highly_effective'
        elif avg_improvement >= 0.05:
            effectiveness_category = 'effective'
        elif avg_improvement >= 0.0:
            effectiveness_category = 'minimally_effective'
        else:
            effectiveness_category = 'ineffective'
        
        # Identify resolved vs new issues
        previous_feedback = state.get('critique_feedback_history', [])
        previous_issues = []
        if previous_feedback:
            previous_issues = previous_feedback[-1].issues_remaining
        
        current_issues = []
        for area, analysis in critique_analysis.get('quality_dimension_analysis', {}).items():
            current_issues.extend(analysis.get('issues', []))
        
        issues_resolved = [issue for issue in previous_issues 
                          if not any(issue in curr for curr in current_issues)]
        new_issues = [issue for issue in current_issues 
                     if not any(issue in prev for prev in previous_issues)]
        
        return {
            'overall_effectiveness': min(1.0, max(0.0, avg_improvement * 2.0 + 0.5)),
            'score_improvements': score_improvements,
            'issues_resolved': issues_resolved,
            'new_issues': new_issues,
            'effectiveness_category': effectiveness_category,
            'total_score_improvement': total_improvement,
            'improvement_trend': 'improving' if avg_improvement > 0.02 else 'stable' if avg_improvement >= -0.02 else 'declining'
        }
    
    async def _generate_targeted_feedback(
        self,
        draft: Draft,
        critique_analysis: Dict[str, Any],
        improvement_assessment: Dict[str, Any],
        cycle_count: int
    ) -> Dict[str, Any]:
        """Generate specific, actionable feedback for the next editing cycle."""
        
        system_prompt = critique_agent_instruction
        
        # Determine feedback focus based on cycle and effectiveness
        if cycle_count == 1:
            feedback_focus = "comprehensive improvement suggestions across all dimensions"
        else:
            if improvement_assessment['effectiveness_category'] == 'ineffective':
                feedback_focus = "alternative approaches for persistent issues"
            else:
                feedback_focus = "refinement and final polish improvements"
        
        human_prompt = f"""Generate targeted feedback based on this critique analysis:

Critique Results:
{json.dumps(critique_analysis, indent=2)}

Improvement Assessment:
- Effectiveness: {improvement_assessment['effectiveness_category']}
- Score Changes: {json.dumps(improvement_assessment['score_improvements'], indent=2)}
- Issues Resolved: {len(improvement_assessment['issues_resolved'])}
- New Issues: {len(improvement_assessment['new_issues'])}

Current Cycle: {cycle_count}
Focus: {feedback_focus}

Return JSON with structured feedback:
{{
    "priority_improvements": [
        {{
            "priority": "critical|high|medium|low",
            "area": "fact_check|domain_expertise|style|compliance|technical",
            "issue": "specific issue description",
            "suggested_action": "concrete action to take",
            "expected_impact": "high|medium|low",
            "implementation_difficulty": "easy|moderate|hard"
        }}
    ],
    "strategic_recommendations": [
        {{
            "recommendation": "strategic improvement suggestion",
            "rationale": "why this will help",
            "implementation_steps": ["step 1", "step 2", "step 3"]
        }}
    ],
    "alternative_approaches": [
        {{
            "current_approach": "what's currently being done",
            "alternative": "different approach to try",
            "potential_benefit": "expected improvement"
        }}
    ],
    "success_metrics": [
        {{
            "metric": "specific measurable outcome",
            "current_value": "current state",
            "target_value": "desired state"
        }}
    ]
}}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            targeted_feedback = json.loads(response.content)
            return targeted_feedback
            
        except Exception as e:
            logger.warning(f"Targeted feedback generation failed: {e}")
            return self._create_fallback_targeted_feedback(critique_analysis)
    
    def _make_retry_decision(
        self,
        critique_analysis: Dict[str, Any],
        improvement_assessment: Dict[str, Any],
        cycle_count: int,
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Make intelligent decision about whether to retry editing."""
        
        overall_score = critique_analysis.get('overall_assessment', {}).get('overall_score_assessment', 0.0)
        meets_standards = critique_analysis.get('publication_readiness', {}).get('ready_to_publish', False)
        critical_issues = critique_analysis.get('overall_assessment', {}).get('critical_issues_count', 0)
        compliance_score = state['quality_scores'].compliance or 0.0
        
        # Decision logic
        if cycle_count >= self.max_critique_cycles:
            # Maximum cycles reached
            if compliance_score < self.critical_compliance_threshold:
                decision = "fail"
                reasoning = f"Maximum critique cycles ({self.max_critique_cycles}) reached with compliance issues (score: {compliance_score:.2f})"
                next_actions = ["human_review", "legal_review", "major_revision"]
            elif overall_score >= self.pass_threshold:
                decision = "pass"
                reasoning = f"Maximum cycles reached but quality threshold met (score: {overall_score:.2f})"
                next_actions = ["proceed_to_formatting"]
            else:
                decision = "fail"
                reasoning = f"Maximum cycles reached without meeting quality threshold (score: {overall_score:.2f})"
                next_actions = ["human_review", "major_revision"]
        
        elif meets_standards and overall_score >= self.pass_threshold and critical_issues == 0:
            # Content meets publication standards
            decision = "pass"
            reasoning = f"Content meets publication standards (score: {overall_score:.2f}, critical issues: {critical_issues})"
            next_actions = ["proceed_to_formatting"]
        
        elif compliance_score < self.critical_compliance_threshold:
            # Critical compliance issues
            decision = "fail"
            reasoning = f"Critical compliance issues detected (compliance score: {compliance_score:.2f})"
            next_actions = ["human_review", "legal_review", "compliance_revision"]
        
        elif critical_issues > 0 or overall_score < 0.70:
            # Significant issues requiring another cycle
            decision = "retry"
            reasoning = f"Significant issues requiring revision (score: {overall_score:.2f}, critical issues: {critical_issues})"
            next_actions = ["focused_editing", "address_critical_issues"]
        
        elif improvement_assessment['effectiveness_category'] in ['ineffective', 'minimally_effective'] and cycle_count > 1:
            # Editing not being effective
            decision = "fail"
            reasoning = "Editing cycles not producing sufficient improvement"
            next_actions = ["human_review", "alternative_editing_strategy"]
        
        elif overall_score < self.pass_threshold:
            # Below threshold but improvable
            decision = "retry"
            reasoning = f"Below quality threshold but showing improvement potential (score: {overall_score:.2f})"
            next_actions = ["targeted_editing", "focus_on_weak_areas"]
        
        else:
            # Edge case - default to pass
            decision = "pass"
            reasoning = "Content quality acceptable with minor issues"
            next_actions = ["proceed_to_formatting", "monitor_quality"]
        
        return {
            'decision': decision,
            'reasoning': reasoning,
            'next_actions': next_actions,
            'confidence': 0.9,  # High confidence in decision logic
            'cycle_count': cycle_count,
            'quality_assessment': {
                'overall_score': overall_score,
                'meets_standards': meets_standards,
                'critical_issues': critical_issues,
                'compliance_score': compliance_score
            }
        }
    
    def _create_structured_feedback(
        self,
        critique_analysis: Dict[str, Any],
        targeted_feedback: Dict[str, Any],
        retry_decision: Dict[str, Any],
        cycle_count: int,
        improvement_assessment: Dict[str, Any]
    ) -> CritiqueFeedback:
        """Create structured critique feedback object for storage and use."""
        
        # Extract issues from critique analysis
        issues_identified = []
        for area, analysis in critique_analysis.get('quality_dimension_analysis', {}).items():
            for issue in analysis.get('issues', []):
                issues_identified.append({
                    'area': area,
                    'issue': issue,
                    'severity': 'critical' if area == 'compliance' else 'moderate',
                    'cycle_identified': cycle_count
                })
        
        # Extract resolved and remaining issues
        issues_resolved = [
            {'issue': issue, 'cycle_resolved': cycle_count}
            for issue in improvement_assessment.get('issues_resolved', [])
        ]
        
        issues_remaining = [
            {'issue': issue['issue'], 'area': issue['area'], 'severity': issue['severity']}
            for issue in issues_identified
        ]
        
        # Extract improvement suggestions
        improvement_suggestions = []
        for improvement in targeted_feedback.get('priority_improvements', []):
            improvement_suggestions.append({
                'suggestion': improvement.get('suggested_action', ''),
                'priority': improvement.get('priority', 'medium'),
                'area': improvement.get('area', 'general'),
                'expected_impact': improvement.get('expected_impact', 'medium'),
                'implementation_difficulty': improvement.get('implementation_difficulty', 'moderate')
            })
        
        return CritiqueFeedback(
            cycle_number=cycle_count,
            overall_score=critique_analysis.get('overall_assessment', {}).get('overall_score_assessment', 0.0),
            issues_identified=issues_identified,
            issues_resolved=issues_resolved,
            issues_remaining=issues_remaining,
            improvement_suggestions=improvement_suggestions,
            retry_decision=retry_decision['decision'],
            retry_reasoning=retry_decision['reasoning'],
            next_actions=retry_decision['next_actions']
        )
    
    async def _collect_agent_learning_data(
        self,
        draft: Draft,
        state: ContentPipelineState,
        critique_feedback: CritiqueFeedback,
        improvement_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect structured learning data for agent improvement."""
        
        learning_data = {
            'pipeline_run_id': state['run_id'],
            'content_metadata': {
                'word_count': draft.word_count,
                'domain_focus': state.get('domain_focus', []),
                'complexity_indicators': {
                    'technical_depth': len([kw for kw in draft.keywords if kw in ['API', 'ML', 'AI', 'algorithm', 'framework']]),
                    'citation_count': len(draft.citations),
                    'section_count': draft.content.count('#')
                }
            },
            'agent_performance': {
                'research_agent': {
                    'source_quality': sum([src.credibility_score for src in state.get('sources', [])]) / max(len(state.get('sources', [])), 1),
                    'source_utilization': len(draft.citations) / max(len(state.get('sources', [])), 1)
                },
                'writing_agent': {
                    'initial_quality': state.get('pre_edit_quality_scores', QualityScores()).calculate_overall(),
                    'content_structure_quality': min(1.0, draft.content.count('#') / 5.0)  # Assume 5 sections is optimal
                },
                'editing_agent': {
                    'improvement_effectiveness': improvement_assessment['overall_effectiveness'],
                    'issue_resolution_rate': len(improvement_assessment['issues_resolved']) / max(len(improvement_assessment['issues_resolved']) + len(improvement_assessment['new_issues']), 1)
                },
                'quality_gates': {
                    'accuracy': self._assess_quality_gate_accuracy(state, critique_feedback),
                    'consistency': self._assess_quality_gate_consistency(state)
                }
            },
            'successful_patterns': self._identify_successful_patterns(state, improvement_assessment),
            'failure_patterns': self._identify_failure_patterns(state, critique_feedback),
            'training_features': {
                'content_features': self._extract_content_features(draft),
                'process_features': self._extract_process_features(state),
                'quality_features': self._extract_quality_features(state['quality_scores'])
            },
            'training_labels': {
                'final_quality_score': critique_feedback.overall_score,
                'improvement_success': improvement_assessment['overall_effectiveness'] > 0.7,
                'retry_required': critique_feedback.retry_decision == 'retry'
            },
            'data_quality_score': self._calculate_learning_data_quality(state, critique_feedback)
        }
        
        return learning_data
    
    def _extract_previous_issues(self, state: ContentPipelineState) -> List[Dict[str, Any]]:
        """Extract issues from previous critique cycles for focused analysis."""
        feedback_history = state.get('critique_feedback_history', [])
        if not feedback_history:
            return []
        
        latest_feedback = feedback_history[-1]
        return latest_feedback.issues_remaining
    
    def _validate_and_enrich_critique(
        self,
        critique_analysis: Dict[str, Any],
        quality_scores: QualityScores
    ) -> Dict[str, Any]:
        """Validate and enrich critique analysis with additional context."""
        
        # Ensure all required fields exist
        if 'overall_assessment' not in critique_analysis:
            critique_analysis['overall_assessment'] = {
                'meets_publication_standards': quality_scores.overall >= 0.85,
                'overall_score_assessment': quality_scores.overall or 0.0,
                'critical_issues_count': 0,
                'moderate_issues_count': 0,
                'minor_issues_count': 0
            }
        
        # Add quality score validation
        critique_analysis['quality_score_validation'] = {
            'scores_realistic': True,  # Could add logic to validate score realism
            'score_consistency': self._check_score_consistency(quality_scores),
            'improvement_potential': max(0.0, 0.95 - (quality_scores.overall or 0.0))
        }
        
        return critique_analysis
    
    def _check_score_consistency(self, quality_scores: QualityScores) -> float:
        """Check consistency between individual scores and overall score."""
        if not quality_scores.overall:
            return 0.0
        
        expected_overall = quality_scores.calculate_overall()
        actual_overall = quality_scores.overall
        
        consistency = 1.0 - abs(expected_overall - actual_overall)
        return max(0.0, consistency)
    
    def _create_fallback_critique_analysis(self, quality_scores: QualityScores) -> Dict[str, Any]:
        """Create fallback critique analysis if AI analysis fails."""
        overall = quality_scores.overall or 0.0
        
        return {
            'overall_assessment': {
                'meets_publication_standards': overall >= 0.85,
                'overall_score_assessment': overall,
                'critical_issues_count': 1 if overall < 0.7 else 0,
                'moderate_issues_count': 1 if 0.7 <= overall < 0.85 else 0,
                'minor_issues_count': 0
            },
            'quality_dimension_analysis': {
                'fact_check': {'score_accurate': True, 'issues': [], 'strengths': []},
                'domain_expertise': {'score_accurate': True, 'issues': [], 'strengths': []},
                'style_consistency': {'score_accurate': True, 'issues': [], 'strengths': []},
                'compliance': {'score_accurate': True, 'issues': [], 'strengths': []},
                'technical_depth': {'score_accurate': True, 'issues': [], 'strengths': []}
            },
            'publication_readiness': {
                'ready_to_publish': overall >= 0.85,
                'blocking_issues': ['AI analysis failed - manual review needed'] if overall < 0.85 else [],
                'recommended_improvements': ['Complete manual review'],
                'estimated_improvement_potential': max(0.0, 0.95 - overall)
            }
        }
    
    def _create_fallback_targeted_feedback(self, critique_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback targeted feedback if AI generation fails."""
        return {
            'priority_improvements': [{
                'priority': 'high',
                'area': 'general',
                'issue': 'Automated feedback generation failed',
                'suggested_action': 'Conduct manual review and editing',
                'expected_impact': 'medium',
                'implementation_difficulty': 'moderate'
            }],
            'strategic_recommendations': [{
                'recommendation': 'Manual quality review',
                'rationale': 'Automated critique system encountered an error',
                'implementation_steps': ['Human review', 'Manual editing', 'Re-assessment']
            }],
            'alternative_approaches': [],
            'success_metrics': [{
                'metric': 'manual_review_completion',
                'current_value': 'not_started',
                'target_value': 'completed'
            }]
        }
    
    def _create_fallback_critique_result(
        self,
        error: Exception,
        cycle_count: int,
        quality_scores: QualityScores
    ) -> Dict[str, Any]:
        """Create fallback critique result when analysis completely fails."""
        return {
            'critique_feedback': CritiqueFeedback(
                cycle_number=cycle_count,
                overall_score=quality_scores.overall or 0.0,
                issues_identified=[{
                    'area': 'system',
                    'issue': f'Critique analysis failed: {str(error)}',
                    'severity': 'critical',
                    'cycle_identified': cycle_count
                }],
                issues_resolved=[],
                issues_remaining=[],
                improvement_suggestions=[{
                    'suggestion': 'Conduct manual review due to system error',
                    'priority': 'critical',
                    'area': 'system',
                    'expected_impact': 'high',
                    'implementation_difficulty': 'moderate'
                }],
                retry_decision='fail',
                retry_reasoning=f'Critique system error: {str(error)}',
                next_actions=['human_review', 'system_check']
            ),
            'retry_decision': 'fail',
            'retry_reasoning': f'System error in critique analysis: {str(error)}',
            'improvement_effectiveness': 0.0,
            'learning_data': {'error': str(error), 'data_quality_score': 0.0},
            'next_actions': ['human_review', 'system_diagnostics']
        }
    
    def _assess_quality_gate_accuracy(
        self, 
        state: ContentPipelineState, 
        critique_feedback: CritiqueFeedback
    ) -> float:
        """Assess how accurate the quality gates were compared to critique analysis."""
        # Placeholder - would implement actual accuracy assessment
        return 0.8
    
    def _assess_quality_gate_consistency(self, state: ContentPipelineState) -> float:
        """Assess consistency of quality gate assessments across cycles."""
        # Placeholder - would implement actual consistency assessment
        return 0.8
    
    def _identify_successful_patterns(
        self, 
        state: ContentPipelineState, 
        improvement_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify patterns associated with successful content creation."""
        # Placeholder - would implement pattern recognition
        return []
    
    def _identify_failure_patterns(
        self, 
        state: ContentPipelineState, 
        critique_feedback: CritiqueFeedback
    ) -> List[Dict[str, Any]]:
        """Identify patterns associated with content issues."""
        # Placeholder - would implement failure pattern recognition  
        return []
    
    def _extract_content_features(self, draft: Draft) -> Dict[str, Any]:
        """Extract features from content for machine learning."""
        return {
            'word_count': draft.word_count,
            'reading_time': draft.reading_time_minutes,
            'citation_count': len(draft.citations),
            'keyword_count': len(draft.keywords),
            'section_count': draft.content.count('#'),
            'paragraph_count': len([p for p in draft.content.split('\n\n') if p.strip()])
        }
    
    def _extract_process_features(self, state: ContentPipelineState) -> Dict[str, Any]:
        """Extract process features for machine learning."""
        return {
            'domain_count': len(state.get('domain_focus', [])),
            'source_count': len(state.get('sources', [])),
            'topic_count': len(state.get('topics', [])),
            'retry_count': state.get('retry_count', 0),
            'step_count': len(state.get('step_history', []))
        }
    
    def _extract_quality_features(self, quality_scores: QualityScores) -> Dict[str, Any]:
        """Extract quality score features for machine learning."""
        return {
            'fact_check_score': quality_scores.fact_check or 0.0,
            'domain_expertise_score': quality_scores.domain_expertise or 0.0,
            'style_consistency_score': quality_scores.style_consistency or 0.0,
            'compliance_score': quality_scores.compliance or 0.0,
            'technical_depth_score': quality_scores.technical_depth or 0.0,
            'overall_score': quality_scores.overall or 0.0,
            'score_variance': self._calculate_score_variance(quality_scores)
        }
    
    def _calculate_score_variance(self, quality_scores: QualityScores) -> float:
        """Calculate variance in quality scores to assess consistency."""
        scores = [
            quality_scores.fact_check or 0.0,
            quality_scores.domain_expertise or 0.0,
            quality_scores.style_consistency or 0.0,
            quality_scores.compliance or 0.0,
            quality_scores.technical_depth or 0.0
        ]
        
        if not scores:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance
    
    def _calculate_learning_data_quality(
        self, 
        state: ContentPipelineState, 
        critique_feedback: CritiqueFeedback
    ) -> float:
        """Calculate the quality of learning data collected for training."""
        quality_factors = []
        
        # Data completeness
        completeness = (
            (1.0 if state.get('sources') else 0.0) +
            (1.0 if state.get('quality_scores') else 0.0) +
            (1.0 if critique_feedback.issues_identified else 0.0) +
            (1.0 if critique_feedback.improvement_suggestions else 0.0)
        ) / 4.0
        quality_factors.append(completeness)
        
        # Process success (successful processes generate better training data)
        process_success = min(1.0, critique_feedback.overall_score)
        quality_factors.append(process_success)
        
        # Feedback richness
        feedback_richness = min(1.0, (
            len(critique_feedback.issues_identified) +
            len(critique_feedback.improvement_suggestions)
        ) / 10.0)
        quality_factors.append(feedback_richness)
        
        return sum(quality_factors) / len(quality_factors)