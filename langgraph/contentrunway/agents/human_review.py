"""Human Review Gate Agent - Manages human review process and feedback collection."""

from typing import Dict, Any, Optional
import logging
import uuid
from datetime import datetime, timedelta
import json

from ..state.pipeline_state import Draft, QualityScores, HumanReviewFeedback, ContentPipelineState

logger = logging.getLogger(__name__)


# Human Review Gate Agent Instructions
human_review_agent_role_and_goal = """
You are a Human Review Gate Agent specializing in managing the 15-minute human review process and feedback collection for content approval workflows.
Your primary goal is to assess review necessity, create structured review sessions, prepare comprehensive review materials, and facilitate efficient human feedback collection for content quality assurance.
"""

human_review_agent_hints = """
Human review management best practices:
- Assess review necessity based on quality scores: optional (≥95%), recommended (≥85%), required (<85%)
- Create time-limited review sessions with 15-minute approval workflow
- Prepare comprehensive review materials including content preview, quality summary, and review checklist
- Focus attention areas on quality dimensions below 80% threshold
- Provide structured review interface with approval, rejection, and revision options
- Generate review checklists prioritized by content accuracy, quality, and compliance
- Estimate review time based on review type and attention areas (3-15 minutes)
- Enable inline editing and provide editing suggestions for reviewer guidance
- Store review session data for tracking and follow-up
"""

human_review_agent_output_description = """
The Human Review Gate Agent returns a comprehensive review session package containing:
- review_url: URL to review interface for human reviewer access
- session_id: Unique identifier for review session tracking
- review_recommendation: Assessment of review necessity with type, urgency, and estimated time
- review_materials: Complete review package with content preview, quality summary, checklist, and editing suggestions
- time_limit_seconds: Review session time limit (900 seconds / 15 minutes)
- expires_at: Session expiration timestamp

The review_materials include content_preview with section summaries, quality_summary with score breakdown, review_checklist with categorized items, and editing_suggestions.
"""

human_review_agent_chain_of_thought_directions = """
Human review management workflow:
1. Use '_assess_review_necessity' to determine review type based on quality scores
2. Apply review type logic: optional (≥95%), recommended (≥85%), required (<85%)
3. Identify attention areas for quality dimensions below 80% threshold
4. Check for critical issues from state critique notes
5. Use '_estimate_review_time' based on review type and attention areas count
6. Use '_create_review_session' to establish review session with unique ID and expiration
7. Apply session expiration with 4x time buffer (60 minutes total)
8. Use '_prepare_review_materials' to create comprehensive review package
9. Apply '_create_content_preview' to generate section summaries and metadata
10. Use '_create_quality_summary' to format quality scores with status indicators
11. Apply '_create_review_checklist' with categorized items (accuracy, quality, compliance)
12. Use '_prepare_editing_suggestions' from critique notes and quality reports
13. Use '_create_review_interface_data' to structure review interface configuration
14. Store review session data using '_store_review_session'
15. Return complete review session package with URL, materials, and configuration

Tool usage conditions:
- Apply review necessity assessment for all content requiring human review
- Create review sessions when review is needed (not auto-approved)
- Generate review materials for all review sessions
- Use content preview creation for content with multiple sections
- Apply quality summary formatting with color-coded status indicators
- Create review checklists based on quality score thresholds
- Store session data for tracking and audit purposes
"""

human_review_agent_instruction = f"""
{human_review_agent_role_and_goal}
{human_review_agent_hints}
{human_review_agent_output_description}
{human_review_agent_chain_of_thought_directions}
"""


class HumanReviewGateAgent:
    """Manages the human review process with 15-minute approval workflow using structured instructions."""
    
    def __init__(self):
        # Review session configuration based on agent instructions
        self.review_time_limit = 15 * 60  # 15 minutes in seconds
        self.auto_approve_threshold = 0.95  # If quality score is this high, suggest auto-approval
        self.requires_review_threshold = 0.85  # Below this score, review is mandatory
        self.agent_instruction = human_review_agent_instruction
        
    async def execute(
        self,
        draft: Draft,
        quality_scores: QualityScores,
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """
        Set up human review session and manage the review process.
        
        Args:
            draft: Content draft for review
            quality_scores: Quality assessment results
            state: Current pipeline state
            
        Returns:
            Dictionary with review session details and process information
        """
        logger.info("Setting up human review session")
        
        try:
            # Step 1: Assess if review is needed
            review_recommendation = self._assess_review_necessity(quality_scores, state)
            
            # Step 2: Create review session
            review_session = self._create_review_session(
                draft, 
                quality_scores, 
                review_recommendation,
                state
            )
            
            # Step 3: Prepare review materials
            review_materials = self._prepare_review_materials(
                draft,
                quality_scores,
                state
            )
            
            # Step 4: Generate review URL/interface data
            review_interface = self._create_review_interface_data(
                review_session,
                review_materials
            )
            
            logger.info(f"Human review session created: {review_session['session_id']}")
            
            return {
                'review_url': review_interface['url'],
                'session_id': review_session['session_id'],
                'review_recommendation': review_recommendation,
                'review_materials': review_materials,
                'time_limit_seconds': self.review_time_limit,
                'expires_at': review_session['expires_at']
            }
            
        except Exception as e:
            logger.error(f"Human review setup failed: {e}")
            # Fallback: create minimal review session
            return self._create_fallback_review_session(draft, state)
    
    def _assess_review_necessity(
        self,
        quality_scores: QualityScores,
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Assess whether human review is necessary and what type."""
        
        overall_score = quality_scores.overall or quality_scores.calculate_overall()
        
        # Determine review necessity
        if overall_score >= self.auto_approve_threshold:
            review_type = 'optional'
            urgency = 'low'
            recommendation = 'Content quality is excellent - review optional'
        elif overall_score >= self.requires_review_threshold:
            review_type = 'recommended'
            urgency = 'medium'
            recommendation = 'Content quality is good - quick review recommended'
        else:
            review_type = 'required'
            urgency = 'high'
            recommendation = 'Content quality needs improvement - thorough review required'
        
        # Identify specific areas needing attention
        attention_areas = []
        
        if quality_scores.fact_check and quality_scores.fact_check < 0.8:
            attention_areas.append('Fact-checking and source verification')
            
        if quality_scores.domain_expertise and quality_scores.domain_expertise < 0.8:
            attention_areas.append('Domain expertise and technical accuracy')
            
        if quality_scores.style_consistency and quality_scores.style_consistency < 0.8:
            attention_areas.append('Writing style and tone consistency')
            
        if quality_scores.compliance and quality_scores.compliance < 0.9:
            attention_areas.append('Legal and ethical compliance')
        
        # Check for critical issues from state
        critical_issues = state.get('critique_notes', [])
        critical_issues = [note for note in critical_issues if 'critical' in note.lower() or 'error' in note.lower()]
        
        if critical_issues:
            review_type = 'required'
            urgency = 'high'
            attention_areas.extend(critical_issues[:3])  # Add top 3 critical issues
        
        return {
            'review_type': review_type,
            'urgency': urgency,
            'recommendation': recommendation,
            'overall_score': overall_score,
            'attention_areas': attention_areas,
            'estimated_review_time': self._estimate_review_time(review_type, attention_areas)
        }
    
    def _estimate_review_time(self, review_type: str, attention_areas: list) -> int:
        """Estimate how long the review should take in minutes."""
        
        base_time = {
            'optional': 3,      # 3 minutes for quick scan
            'recommended': 8,   # 8 minutes for thorough review
            'required': 15      # 15 minutes for detailed review
        }
        
        estimated_time = base_time.get(review_type, 10)
        
        # Add time for each attention area
        estimated_time += len(attention_areas) * 2
        
        return min(15, estimated_time)  # Cap at 15 minutes
    
    def _create_review_session(
        self,
        draft: Draft,
        quality_scores: QualityScores,
        review_recommendation: Dict[str, Any],
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Create a new human review session."""
        
        session_id = str(uuid.uuid4())
        created_at = datetime.now()
        expires_at = created_at + timedelta(seconds=self.review_time_limit * 4)  # Give extra time buffer
        
        session = {
            'session_id': session_id,
            'run_id': state['run_id'],
            'created_at': created_at.isoformat(),
            'expires_at': expires_at.isoformat(),
            'status': 'pending',
            'review_type': review_recommendation['review_type'],
            'estimated_time_minutes': review_recommendation['estimated_review_time'],
            'content_metadata': {
                'title': draft.title,
                'word_count': draft.word_count,
                'reading_time': draft.reading_time_minutes,
                'domain_focus': state.get('domain_focus', [])
            }
        }
        
        # Store session data (in real implementation, this would go to database)
        self._store_review_session(session)
        
        return session
    
    def _prepare_review_materials(
        self,
        draft: Draft,
        quality_scores: QualityScores,
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Prepare all materials needed for human review."""
        
        # Prepare content preview
        content_preview = self._create_content_preview(draft)
        
        # Prepare quality summary
        quality_summary = self._create_quality_summary(quality_scores)
        
        # Prepare review checklist
        review_checklist = self._create_review_checklist(quality_scores, state)
        
        # Prepare editing suggestions
        editing_suggestions = self._prepare_editing_suggestions(state)
        
        return {
            'content_preview': content_preview,
            'quality_summary': quality_summary,
            'review_checklist': review_checklist,
            'editing_suggestions': editing_suggestions,
            'original_research_query': state.get('research_query', ''),
            'target_domains': state.get('domain_focus', []),
            'quality_thresholds': state.get('quality_thresholds', {})
        }
    
    def _create_content_preview(self, draft: Draft) -> Dict[str, Any]:
        """Create a preview of content for review interface."""
        
        # Extract key sections
        content_lines = draft.content.split('\n')
        sections = []
        current_section = None
        
        for line in content_lines:
            if line.startswith('#'):
                if current_section:
                    sections.append(current_section)
                current_section = {'header': line.strip(), 'content': []}
            elif current_section and line.strip():
                current_section['content'].append(line.strip())
        
        if current_section:
            sections.append(current_section)
        
        # Create section summaries
        section_summaries = []
        for section in sections[:5]:  # Limit to first 5 sections
            content_preview = ' '.join(section['content'][:2])  # First 2 sentences/lines
            if len(content_preview) > 150:
                content_preview = content_preview[:150] + '...'
                
            section_summaries.append({
                'header': section['header'],
                'preview': content_preview,
                'word_count': len(' '.join(section['content']).split())
            })
        
        return {
            'title': draft.title,
            'subtitle': draft.subtitle,
            'abstract': draft.abstract,
            'total_word_count': draft.word_count,
            'reading_time_minutes': draft.reading_time_minutes,
            'section_count': len(sections),
            'section_summaries': section_summaries,
            'citations_count': len(draft.citations),
            'keywords': draft.keywords[:10],  # Top 10 keywords
            'tags': draft.tags
        }
    
    def _create_quality_summary(self, quality_scores: QualityScores) -> Dict[str, Any]:
        """Create a summary of quality assessment results."""
        
        overall_score = quality_scores.overall or quality_scores.calculate_overall()
        
        # Convert scores to percentage and status
        def score_to_status(score):
            if score is None:
                return {'percentage': 0, 'status': 'not_assessed', 'color': 'gray'}
            elif score >= 0.9:
                return {'percentage': int(score * 100), 'status': 'excellent', 'color': 'green'}
            elif score >= 0.8:
                return {'percentage': int(score * 100), 'status': 'good', 'color': 'blue'}
            elif score >= 0.7:
                return {'percentage': int(score * 100), 'status': 'acceptable', 'color': 'yellow'}
            else:
                return {'percentage': int(score * 100), 'status': 'needs_improvement', 'color': 'red'}
        
        quality_breakdown = {
            'overall': score_to_status(overall_score),
            'fact_check': score_to_status(quality_scores.fact_check),
            'domain_expertise': score_to_status(quality_scores.domain_expertise),
            'style_consistency': score_to_status(quality_scores.style_consistency),
            'compliance': score_to_status(quality_scores.compliance),
            'technical_depth': score_to_status(quality_scores.technical_depth)
        }
        
        # Identify strengths and concerns
        strengths = []
        concerns = []
        
        for area, status in quality_breakdown.items():
            if area == 'overall':
                continue
                
            if status['status'] == 'excellent':
                strengths.append(f"{area.replace('_', ' ').title()}: {status['percentage']}%")
            elif status['status'] in ['needs_improvement', 'acceptable']:
                concerns.append(f"{area.replace('_', ' ').title()}: {status['percentage']}%")
        
        return {
            'overall_score_percentage': int(overall_score * 100),
            'overall_status': score_to_status(overall_score)['status'],
            'quality_breakdown': quality_breakdown,
            'strengths': strengths,
            'concerns': concerns,
            'meets_quality_threshold': overall_score >= 0.85
        }
    
    def _create_review_checklist(
        self,
        quality_scores: QualityScores,
        state: ContentPipelineState
    ) -> List[Dict[str, Any]]:
        """Create a checklist for human reviewers."""
        
        checklist = []
        
        # Content accuracy checklist
        checklist.append({
            'category': 'Content Accuracy',
            'priority': 'high',
            'items': [
                {
                    'item': 'Factual claims are accurate and properly cited',
                    'focus': quality_scores.fact_check and quality_scores.fact_check < 0.8,
                    'guidance': 'Verify statistics, dates, and technical claims against sources'
                },
                {
                    'item': 'Technical information is current and correct',
                    'focus': quality_scores.domain_expertise and quality_scores.domain_expertise < 0.8,
                    'guidance': 'Check domain-specific terminology and concepts'
                },
                {
                    'item': 'Citations are properly formatted and relevant',
                    'focus': True,
                    'guidance': 'Ensure all claims are supported by credible sources'
                }
            ]
        })
        
        # Content quality checklist
        checklist.append({
            'category': 'Content Quality',
            'priority': 'medium',
            'items': [
                {
                    'item': 'Writing is clear, engaging, and well-structured',
                    'focus': quality_scores.style_consistency and quality_scores.style_consistency < 0.8,
                    'guidance': 'Check for flow, readability, and audience appropriateness'
                },
                {
                    'item': 'Content provides actionable value to readers',
                    'focus': True,
                    'guidance': 'Ensure practical insights and takeaways are included'
                },
                {
                    'item': 'Tone and style are consistent throughout',
                    'focus': quality_scores.style_consistency and quality_scores.style_consistency < 0.8,
                    'guidance': 'Check for voice consistency and professional tone'
                }
            ]
        })
        
        # Compliance checklist
        checklist.append({
            'category': 'Legal & Ethical Compliance',
            'priority': 'high',
            'items': [
                {
                    'item': 'No copyright or intellectual property violations',
                    'focus': quality_scores.compliance and quality_scores.compliance < 0.9,
                    'guidance': 'Check for proper attribution and fair use'
                },
                {
                    'item': 'Content is free from bias and discriminatory language',
                    'focus': quality_scores.compliance and quality_scores.compliance < 0.9,
                    'guidance': 'Review for inclusive language and fair representation'
                },
                {
                    'item': 'Professional and ethical standards are maintained',
                    'focus': True,
                    'guidance': 'Ensure content meets industry ethical guidelines'
                }
            ]
        })
        
        return checklist
    
    def _prepare_editing_suggestions(self, state: ContentPipelineState) -> List[Dict[str, Any]]:
        """Prepare editing suggestions from quality feedback."""
        
        critique_notes = state.get('critique_notes', [])
        suggestions = []
        
        for i, note in enumerate(critique_notes[:8]):  # Limit to 8 suggestions
            suggestions.append({
                'id': f'suggestion_{i+1}',
                'type': 'improvement',
                'description': note,
                'priority': 'medium',
                'action': 'Consider implementing this improvement'
            })
        
        # Add any specific recommendations from quality reports
        fact_check_report = state.get('fact_check_report', {})
        if isinstance(fact_check_report, dict) and 'recommendations' in fact_check_report:
            for rec in fact_check_report['recommendations'][:3]:
                suggestions.append({
                    'id': f'fact_check_{len(suggestions)}',
                    'type': 'fact_check',
                    'description': rec,
                    'priority': 'high',
                    'action': 'Address this fact-checking concern'
                })
        
        return suggestions
    
    def _create_review_interface_data(
        self,
        review_session: Dict[str, Any],
        review_materials: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create data structure for review interface."""
        
        # In a real implementation, this would generate a URL to a web interface
        # For now, we'll create a structured data representation
        
        interface_url = f"/review/{review_session['session_id']}"
        
        return {
            'url': interface_url,
            'session_data': review_session,
            'materials': review_materials,
            'actions': {
                'approve': {'endpoint': f"{interface_url}/approve", 'method': 'POST'},
                'reject': {'endpoint': f"{interface_url}/reject", 'method': 'POST'},
                'request_revision': {'endpoint': f"{interface_url}/revise", 'method': 'POST'}
            },
            'interface_config': {
                'show_content_preview': True,
                'show_quality_scores': True,
                'show_editing_suggestions': True,
                'enable_inline_editing': True,
                'time_limit_warning': True
            }
        }
    
    def _store_review_session(self, session: Dict[str, Any]):
        """Store review session data (placeholder for database storage)."""
        # In a real implementation, this would store to database
        logger.info(f"Review session stored: {session['session_id']}")
        pass
    
    def _create_fallback_review_session(
        self,
        draft: Draft,
        state: ContentPipelineState
    ) -> Dict[str, Any]:
        """Create a fallback review session if setup fails."""
        
        session_id = str(uuid.uuid4())
        
        return {
            'review_url': f"/review/{session_id}",
            'session_id': session_id,
            'review_recommendation': {
                'review_type': 'recommended',
                'urgency': 'medium',
                'recommendation': 'Content review recommended',
                'overall_score': 0.8,
                'attention_areas': ['General content quality'],
                'estimated_review_time': 10
            },
            'time_limit_seconds': self.review_time_limit,
            'expires_at': (datetime.now() + timedelta(seconds=self.review_time_limit * 4)).isoformat(),
            'error': 'Fallback review session created due to setup error'
        }
    
    def process_review_feedback(
        self,
        session_id: str,
        feedback_data: Dict[str, Any]
    ) -> HumanReviewFeedback:
        """Process feedback submitted through the review interface."""
        
        # Validate feedback data
        decision = feedback_data.get('decision', 'needs_revision')
        if decision not in ['approved', 'rejected', 'needs_revision']:
            decision = 'needs_revision'
        
        # Create HumanReviewFeedback object
        feedback = HumanReviewFeedback(
            decision=decision,
            overall_rating=feedback_data.get('overall_rating'),
            feedback_notes=feedback_data.get('feedback_notes'),
            inline_edits=feedback_data.get('inline_edits', []),
            quality_concerns=feedback_data.get('quality_concerns', []),
            time_spent_seconds=feedback_data.get('time_spent_seconds')
        )
        
        logger.info(f"Review feedback processed for session {session_id}: {decision}")
        
        return feedback
    
    def get_review_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a review session."""
        
        # In real implementation, this would query database
        return {
            'session_id': session_id,
            'status': 'pending',  # pending, completed, expired
            'time_remaining_seconds': 900,  # Example: 15 minutes
            'last_activity': datetime.now().isoformat()
        }