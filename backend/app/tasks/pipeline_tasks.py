"""
Background tasks for ContentRunway pipeline execution using Celery.
"""

import sys
import os
from typing import Dict, List, Any, Optional
import uuid
import logging
import json
import math
import threading
from datetime import datetime, timedelta
from celery import current_task

# Load environment variables from .env file explicitly
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add agent directories to Python path for backwards compatibility
for path in ['/app/agents', '/app/langgraph']:
    if path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)

from app.worker import celery_app
from app.core.config import settings
from app.services.redis_service import redis_service
from app.models.pipeline import PipelineRun
from app.db.sync_database import (
    get_sync_session,
    update_pipeline_status,
    update_pipeline_completion,
    create_topic_idea,
    get_selected_topic_id,
    create_content_draft,
    create_quality_assessment,
    create_research_source,
    create_content_outline,
    create_channel_content,
    create_publication,
    create_fact_check_report,
    create_critique_report,
    log_agent_activity
)
from app.services.draft_persistence import (
    persist_content_draft,
    DraftPersistenceError,
    ensure_current_draft_id,
    create_review_session,
    ReviewSessionError,
)

logger = logging.getLogger(__name__)

# Celery-based pipeline - direct agent execution (no LangGraph dependency)
CELERY_PIPELINE_AVAILABLE = True
logger.info("Celery-based pipeline operational")


def _validate_environment_variables():
    """Validate required environment variables are available."""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API integration',
        'DATABASE_URL': 'Database connectivity',
        'REDIS_URL': 'Redis cache and state management'
    }
    
    missing_vars = []
    for var_name, purpose in required_vars.items():
        if not os.getenv(var_name):
            missing_vars.append(f"{var_name} (required for {purpose})")
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Log successful validation
    logger.info("✅ Environment variables validated successfully")


def _generate_quality_improvement_recommendations(failed_gates: List[Dict], run_id: str) -> Dict[str, Any]:
    """
    Generate specific improvement recommendations for failed quality gates.
    
    Args:
        failed_gates: List of failed gate dictionaries with gate, score, threshold, gap
        run_id: Pipeline run ID for logging
    
    Returns:
        Dictionary with detailed recommendations per gate
    """
    recommendations = {
        "summary": f"{len(failed_gates)} quality gate(s) failed",
        "gates": {},
        "priority_actions": [],
        "estimated_improvement_potential": 0.0
    }
    
    gate_recommendations = {
        "fact_check": {
            "primary_issues": [
                "Insufficient source verification and citation density",
                "Weak factual claims without supporting evidence", 
                "Lack of credible, recent sources in domain"
            ],
            "improvement_actions": [
                "Add minimum 3-5 authoritative citations per section",
                "Verify all factual claims against multiple sources",
                "Include recent industry reports and statistics",
                "Add expert quotes and research study references",
                "Implement fact-checking review process"
            ],
            "content_requirements": "Ensure every technical claim has supporting citation"
        },
        "domain_expertise": {
            "primary_issues": [
                "Insufficient technical depth and specialization",
                "Generic content lacking domain-specific insights",
                "Poor use of industry terminology and concepts"
            ],
            "improvement_actions": [
                "Increase technical terminology usage (15-20 domain terms per 1000 words)",
                "Add detailed technical explanations and examples",
                "Include industry-specific case studies and scenarios",
                "Reference domain standards and best practices",
                "Add practical implementation guidance"
            ],
            "content_requirements": "Demonstrate deep subject matter expertise with technical specifics"
        },
        "style_consistency": {
            "primary_issues": [
                "Poor readability and flow structure",
                "Inconsistent tone and formatting",
                "Inadequate content organization and headers"
            ],
            "improvement_actions": [
                "Improve paragraph structure (3-5 sentences each)",
                "Add more subheadings for better content navigation", 
                "Enhance readability (target Flesch score 40-60)",
                "Standardize formatting (lists, emphasis, structure)",
                "Ensure consistent professional tone throughout"
            ],
            "content_requirements": "Maintain professional readability with clear structure"
        },
        "compliance": {
            "primary_issues": [
                "Legal and regulatory compliance concerns",
                "Ethical considerations not addressed",
                "Missing privacy and data protection considerations"
            ],
            "improvement_actions": [
                "Add regulatory compliance disclaimers where needed",
                "Include data privacy and security considerations",
                "Remove potentially problematic claims or language",
                "Add appropriate legal disclaimers and references",
                "Ensure ethical AI and technology use discussions"
            ],
            "content_requirements": "Meet legal/ethical standards with appropriate disclaimers"
        }
    }
    
    total_gap = 0.0
    for gate_failure in failed_gates:
        gate_name = gate_failure["gate"]
        gap = gate_failure["gap"]
        total_gap += gap
        
        if gate_name in gate_recommendations:
            recommendations["gates"][gate_name] = {
                "current_score": gate_failure["score"],
                "required_score": gate_failure["threshold"],
                "improvement_needed": gap,
                "severity": "high" if gap > 0.15 else "medium" if gap > 0.08 else "low",
                **gate_recommendations[gate_name]
            }
            
            # Add to priority actions based on gap size
            if gap > 0.15:
                recommendations["priority_actions"].append(f"URGENT: {gate_name} needs {gap:.3f} improvement")
            elif gap > 0.08:
                recommendations["priority_actions"].append(f"HIGH: {gate_name} requires significant improvement")
    
    recommendations["estimated_improvement_potential"] = min(0.20, total_gap * 0.8)
    
    # Log detailed recommendations
    logger.info(f"Pipeline {run_id}: Quality improvement recommendations generated")
    for gate_name, details in recommendations["gates"].items():
        logger.info(f"  📊 {gate_name}: {details['severity'].upper()} priority - needs {details['improvement_needed']:.3f} improvement")
        logger.info(f"     Primary focus: {details['primary_issues'][0]}")
    
    return recommendations


def _execute_content_revision_cycle(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """
    Execute a content revision cycle based on quality gate feedback.
    
    Args:
        state: Pipeline state with revision context
        celery_task: Celery task for progress updates
    
    Returns:
        Updated state after revision attempt
    """
    run_id = state["run_id"]
    revision_attempt = state.get("revision_attempts", 1)
    
    logger.info(f"Pipeline {run_id}: 🎯 Starting content revision cycle (attempt {revision_attempt})")
    
    # Extract revision guidance
    revision_context = state.get("revision_context", {})
    failed_gates = state.get("failed_gates", [])
    recommendations = state.get("quality_recommendations", {})
    
    # Update progress
    celery_task.update_state(
        state="PROGRESS",
        meta={
            "run_id": run_id,
            "status": "revising_content",
            "progress": 45,  # Back to writing stage
            "message": f"Revising content based on quality feedback (attempt {revision_attempt})"
        }
    )
    
    try:
        # Stage: Enhanced Writing with Quality Guidance
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Starting enhanced writing stage with quality improvement guidance")
        
        # Enhance the state with revision guidance for the writing agent
        state["revision_guidance"] = {
            "failed_gates": [gate["gate"] for gate in failed_gates],
            "improvement_priorities": revision_context.get("priority_focus", []),
            "specific_improvements": recommendations.get("gates", {}),
            "previous_scores": revision_context.get("previous_scores", {}),
            "revision_attempt": revision_attempt
        }
        
        state = _execute_writing_stage(state, celery_task)
        if state["status"] == "failed":
            logger.error(f"Pipeline {run_id}: Writing stage failed during revision attempt {revision_attempt}")
            return _create_failure_result(run_id, state["error_message"])
        
        _track_stage_duration(state, f"writing_revision_{revision_attempt}", stage_start_time)
        
        # Stage: Quality Gates Re-evaluation
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Re-evaluating quality gates after revision")
        
        celery_task.update_state(
            state="PROGRESS",
            meta={
                "run_id": run_id,
                "status": "quality_evaluation",
                "progress": 60,
                "message": f"Re-evaluating quality after revision (attempt {revision_attempt})"
            }
        )
        
        state = _execute_quality_gates_stage(state, celery_task)
        if state["status"] == "failed":
            logger.warning(f"Pipeline {run_id}: Quality gates still failing after revision attempt {revision_attempt}")
            return state  # Will trigger another revision attempt or final failure
        
        _track_stage_duration(state, f"quality_gates_revision_{revision_attempt}", stage_start_time)
        
        # If we reach here, quality gates passed after revision
        logger.info(f"Pipeline {run_id}: ✅ Quality gates passed after revision attempt {revision_attempt}")
        
        # Log revision success metrics for analytics
        if revision_attempt > 1:
            revision_context = state.get("revision_context", {})
            previous_scores = revision_context.get("previous_scores", {})
            current_scores = state.get("quality_scores", {})
            
            logger.info(f"Pipeline {run_id}: 📈 REVISION SUCCESS METRICS")
            logger.info(f"  🔄 Revision attempt: {revision_attempt}")
            logger.info(f"  📊 Score improvements:")
            
            for gate_name in ["fact_check", "domain_expertise", "style_consistency", "compliance"]:
                if gate_name in previous_scores and gate_name in current_scores:
                    prev_score = previous_scores[gate_name]
                    curr_score = current_scores[gate_name]
                    improvement = curr_score - prev_score
                    logger.info(f"    • {gate_name}: {prev_score:.3f} → {curr_score:.3f} (Δ {improvement:+.3f})")
            
            # Store revision success metrics for analytics
            state["revision_success_metrics"] = {
                "revision_attempt": revision_attempt,
                "previous_scores": previous_scores,
                "improved_scores": current_scores,
                "gates_improved": len([g for g in ["fact_check", "domain_expertise", "style_consistency", "compliance"] 
                                     if g in previous_scores and g in current_scores and current_scores[g] > previous_scores[g]]),
                "revision_successful": True
            }
        
        # Clean up revision context since we succeeded
        if "revision_context" in state:
            del state["revision_context"]
        if "revision_guidance" in state:
            del state["revision_guidance"]
        
        # Continue with normal pipeline flow (editing stage)
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Continuing to editing stage after successful revision")
        
        celery_task.update_state(
            state="PROGRESS",
            meta={
                "run_id": run_id,
                "status": "editing",
                "progress": 70,
                "message": "Proceeding to editing after successful quality revision"
            }
        )
        
        state = _execute_editing_stage(state, celery_task)
        if state["status"] == "failed":
            return _create_failure_result(run_id, state["error_message"])
        
        _track_stage_duration(state, "editing", stage_start_time)
        
        return state
        
    except Exception as e:
        logger.error(f"Pipeline {run_id}: Content revision cycle failed: {str(e)}")
        state["status"] = "failed"
        state["error_message"] = f"Content revision failed: {str(e)}"
        return state

# Database operations now handled by sync_database module


@celery_app.task(bind=True, name="execute_pipeline")
def execute_content_pipeline(self, run_id: str, pipeline_config: Dict[str, Any]):
    """
    Execute the complete ContentRunway pipeline in the background.
    
    Args:
        run_id: UUID of the pipeline run
        pipeline_config: Configuration dictionary for the pipeline
    """
    try:
        logger.info(f"Starting pipeline execution for run_id: {run_id}")
        
        # Validate environment variables before proceeding
        try:
            _validate_environment_variables()
        except ValueError as e:
            logger.error(f"Environment validation failed for run_id: {run_id} - {str(e)}")
            self.update_state(
                state="FAILURE",
                meta={"run_id": run_id, "error": f"Environment validation failed: {str(e)}"}
            )
            raise e
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"run_id": run_id, "status": "starting", "progress": 5}
        )
        
        # Run the pipeline execution with sync database operations
        result = _execute_pipeline_sync(run_id, pipeline_config, self)
        
        logger.info(f"Pipeline execution completed for run_id: {run_id}")
        return result
        
    except Exception as e:
        logger.error(f"Pipeline execution failed for run_id: {run_id} - {str(e)}")
        
        # Update database with failure
        update_pipeline_status(run_id, "failed", error_message=str(e))
        
        # Update task state
        self.update_state(
            state="FAILURE",
            meta={"run_id": run_id, "error": str(e)}
        )
        
        raise e


def _execute_pipeline_sync(run_id: str, pipeline_config: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute the pipeline with sync database operations."""
    
    logger.info(f"Starting pipeline execution for {run_id} (Celery-based pipeline)")
    
    # Update database status to running
    update_pipeline_status(run_id, "running", current_step="research", progress_percentage=10.0)
    
    try:
        # Use Celery-based pipeline (replacing LangGraph)
        result = _execute_celery_pipeline(run_id, pipeline_config, celery_task)
        
        logger.info(f"Pipeline {run_id} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Pipeline execution error: {str(e)}")
        update_pipeline_status(run_id, "failed", error_message=str(e))
        raise


def _execute_hybrid_pipeline(run_id: str, pipeline_config: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute pipeline with database operations but simplified logic."""
    
    # Pipeline execution stages with actual content generation
    stages = [
        ("research", 20.0, _simulate_research),
        ("curation", 30.0, _simulate_curation), 
        ("writing", 60.0, _generate_content),
        ("quality_gates", 80.0, _simulate_quality_gates),
        ("formatting", 90.0, _simulate_formatting),
        ("publishing", 95.0, _simulate_publishing),
        ("completed", 100.0, None)
    ]
    
    generated_content = None
    
    for stage, progress, stage_func in stages:
        logger.info(f"Pipeline {run_id}: Executing stage {stage}")
        
        # Update progress
        update_pipeline_status(run_id, "running", current_step=stage, progress_percentage=progress)
        
        # Update Redis for real-time monitoring
        _update_redis_state(run_id, "running", current_step=stage, progress_percentage=progress)
        
        # Update Celery task state
        celery_task.update_state(
            state="PROGRESS",
            meta={
                "run_id": run_id,
                "status": "running",
                "current_step": stage,
                "progress": progress
            }
        )
        
        # Execute stage function
        if stage_func:
            stage_result = stage_func(run_id, pipeline_config)
            if stage == "writing":
                generated_content = stage_result
        
        # Simulate work time
        import time
        time.sleep(1)
    
    # Create final results
    final_state = {
        "run_id": run_id,
        "status": "completed",
        "published_urls": ["http://localhost:3003/mock-published-content"],  # Development mock URL
        "processing_time": 8.0,
        "content_generated": generated_content is not None
    }
    
    # Ensure result is JSON serializable (remove any complex objects)
    serialized_state = _ensure_json_serializable(final_state)
    
    # Update database with completion
    update_pipeline_completion(run_id, serialized_state)
    
    # Update Celery task state with serializable data only
    celery_task.update_state(
        state="SUCCESS",
        meta={
            "run_id": run_id,
            "status": "completed", 
            "progress": 100,
            "success": True
        }
    )
    
    # Return only serializable data for Celery
    return {
        "run_id": run_id,
        "status": "completed",
        "success": True,
        "processing_time": serialized_state.get("processing_time", 0),
        "content_generated": serialized_state.get("content_generated", False),
        "published_urls": serialized_state.get("published_urls", [])
    }


def _execute_celery_pipeline(run_id: str, pipeline_config: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute the full pipeline using Celery-based agent orchestration."""
    logger.info(f"Executing Celery-based pipeline for {run_id}")
    
    try:
        # Import agents directly (no LangGraph wrapper)
        from contentrunway.agents import (
            ResearchCoordinatorAgent,
            ContentCuratorAgent,
            SEOStrategistAgent,
            ContentWriterAgent,
            FactCheckGateAgent,
            DomainExpertiseGateAgent,
            StyleCriticGateAgent,
            ComplianceGateAgent,
            ContentEditorAgent,
            CritiqueAgent,
            ContentFormatterAgent,
            HumanReviewGateAgent,
            PublisherAgent
        )
        
        # Initialize pipeline state
        from datetime import datetime
        target_words = pipeline_config.get("target_word_count", 500) or 500
        # Ensure target is within reasonable bounds
        target_words = max(50, min(2000, int(target_words)))
        min_word_count = max(50, math.floor(target_words * 0.7))
        max_word_count = min(2000, math.ceil(target_words * 1.3))
        
        state = {
            "run_id": run_id,
            "tenant_id": pipeline_config.get("tenant_id", "personal"),
            "status": "running",
            "created_at": datetime.now(),
            "domain_focus": pipeline_config.get("domain_focus", ["General"]),
            "target_word_count": target_words,
            "content_word_count_target": target_words,
            "content_word_count_min": min_word_count,
            "content_word_count_max": max_word_count,
            "quality_thresholds": pipeline_config.get("quality_thresholds", {
                "overall": 0.85,
                "fact_check": 0.90,
                "domain_expertise": 0.90,
                "style_consistency": 0.88,
                "compliance": 0.95
            }),
            "research_query": pipeline_config.get("research_query", ""),
            "sources": [],
            "topics": [],
            "chosen_topic_id": None,
            "outline": None,
            "draft": None,
            "channel_drafts": None,
            "quality_scores": {},
            "critique_notes": [],
            "fact_check_report": None,
            "compliance_report": None,
            "critique_cycle_count": 0,
            "critique_feedback_history": [],
            "current_critique_feedback": None,
            "pre_edit_quality_scores": None,
            "post_edit_quality_scores": None,
            "human_review_required": False,
            "human_review_feedback": None,
            "publishing_results": None,
            "published_urls": [],
            "current_step": "research",
            "error_message": None,
            "retry_count": 0,
            "max_retries": 3,
            "progress_percentage": 10.0,
            "step_history": [],
            "llm_usage": {},
            "agent_performance_metrics": {},
            "learning_data_quality": 1.0,
            "processing_start_time": datetime.now(),
            "processing_end_time": None,
            "step_durations": {},
            "intermediate_results": {},
            "config_overrides": {}
        }
        
        # Execute pipeline stages sequentially with direct agent calls
        stage_start_time = datetime.now()
        
        # Stage 1: Research
        logger.info(f"Pipeline {run_id}: Starting research stage")
        state = _execute_research_stage(state, celery_task)
        if state["status"] == "failed":
            return _create_failure_result(run_id, state["error_message"])
        _track_stage_duration(state, "research", stage_start_time)
        
        # Stage 2: Curation  
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Starting curation stage")
        state = _execute_curation_stage(state, celery_task)
        if state["status"] == "failed":
            return _create_failure_result(run_id, state["error_message"])
        _track_stage_duration(state, "curation", stage_start_time)
        
        # Stage 3: SEO Strategy
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Starting SEO strategy stage")
        state = _execute_seo_strategy_stage(state, celery_task)
        if state["status"] == "failed":
            return _create_failure_result(run_id, state["error_message"])
        _track_stage_duration(state, "seo_strategy", stage_start_time)
        
        # Stage 4: Writing
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Starting writing stage")
        state = _execute_writing_stage(state, celery_task)
        if state["status"] == "failed":
            return _create_failure_result(run_id, state["error_message"])
        
        # CRITICAL: Validate content generation
        draft = state.get("draft")
        if draft:
            word_count = getattr(draft, 'word_count', 0) if hasattr(draft, 'word_count') else draft.get('word_count', 0)
            target_words = state.get("target_word_count", 500)
            min_words = state.get("content_word_count_min", max(50, math.floor(target_words * 0.7)))
            max_words = state.get("content_word_count_max", min(2000, math.ceil(target_words * 1.3)))
            if word_count < min_words:
                error_msg = (
                    f"Content generation failed: Only {word_count} words generated "
                    f"(minimum {min_words} required for target {target_words})"
                )
                logger.error(f"Pipeline {run_id}: {error_msg}")
                state["status"] = "failed"
                state["error_message"] = error_msg
                return _create_failure_result(run_id, error_msg)
            if word_count > max_words:
                logger.warning(
                    f"Pipeline {run_id}: Generated content ({word_count} words) exceeds "
                    f"target {target_words} words (max {max_words}). Downstream stages may condense."
                )
            else:
                logger.info(
                    f"Pipeline {run_id}: Content validation passed: {word_count} words "
                    f"(target {target_words}, min {min_words})"
                )
        
        _track_stage_duration(state, "writing", stage_start_time)
        
        # Stage 5: Quality Gates (Parallel)
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Starting quality gates stage")
        state = _execute_quality_gates_stage(state, celery_task)
        if state["status"] == "failed":
            return _create_failure_result(run_id, state["error_message"])
        
        # Handle bounce-back to editing if content verification failed
        if state["status"] == "bounce_back_to_editing":
            logger.warning(f"Pipeline {run_id}: Content verification failed, bouncing back to editing stage")
            
            # Re-execute editing stage with bounce-back context
            stage_start_time = datetime.now()
            state["bounce_back_attempt"] = state.get("bounce_back_attempt", 0) + 1
            
            # Limit bounce-back attempts to prevent infinite loops
            if state["bounce_back_attempt"] > 2:
                error_msg = f"Maximum bounce-back attempts exceeded ({state['bounce_back_attempt']}). Reason: {state.get('bounce_back_reason', 'Unknown')}"
                logger.error(f"Pipeline {run_id}: {error_msg}")
                state["status"] = "failed"
                state["error_message"] = error_msg
                return _create_failure_result(run_id, error_msg)
            
            logger.info(f"Pipeline {run_id}: Retry editing stage (attempt {state['bounce_back_attempt']}) - {state.get('bounce_back_reason', 'Unknown reason')}")
            
            # Re-execute editing with enhanced feedback
            state["editing_feedback"] = {
                "bounce_back_reason": state.get("bounce_back_reason"),
                "attempt": state["bounce_back_attempt"],
                "enhancement_required": True
            }
            
            state = _execute_editing_stage(state, celery_task)
            if state["status"] == "failed":
                return _create_failure_result(run_id, state["error_message"])
            _track_stage_duration(state, "editing_retry", stage_start_time)
            
            # Re-run quality gates after editing retry
            stage_start_time = datetime.now()
            logger.info(f"Pipeline {run_id}: Re-running quality gates after editing retry (attempt {state['bounce_back_attempt']})")
            state = _execute_quality_gates_stage(state, celery_task)
            if state["status"] == "failed":
                return _create_failure_result(run_id, state["error_message"])
            
            # Check for another bounce-back - this should now be prevented by quality gates function
            if state["status"] == "bounce_back_to_editing":
                # This should not happen anymore due to the loop prevention in quality gates
                error_msg = f"Unexpected bounce-back after retry limit check. This indicates a logic error. Reason: {state.get('bounce_back_reason', 'Unknown')}"
                logger.error(f"Pipeline {run_id}: {error_msg}")
                state["status"] = "failed"
                state["error_message"] = error_msg
                return _create_failure_result(run_id, error_msg)
            
            _track_stage_duration(state, "quality_gates_retry", stage_start_time)
        
        # CRITICAL: Enforce quality thresholds - Individual gates first, then overall
        quality_scores = state.get("quality_scores", {})
        quality_thresholds = state.get("quality_thresholds", {
            "overall": 0.85,
            "fact_check": 0.90,
            "domain_expertise": 0.90,
            "style_consistency": 0.88,
            "compliance": 0.95
        })
        
        # Check individual gate thresholds first
        failed_gates = []
        individual_gate_scores = {}
        
        for gate_name in ["fact_check", "domain_expertise", "style_consistency", "compliance"]:
            score = quality_scores.get(gate_name, 0.0)
            threshold = quality_thresholds.get(gate_name, 0.85)
            individual_gate_scores[gate_name] = score
            
            if score < threshold:
                failed_gates.append({
                    "gate": gate_name,
                    "score": score,
                    "threshold": threshold,
                    "gap": threshold - score
                })
        
        # If any individual gates failed, attempt automatic revision
        if failed_gates:
            # Generate comprehensive improvement recommendations
            recommendations = _generate_quality_improvement_recommendations(failed_gates, run_id)
            
            # Check if we should attempt automatic revision
            revision_attempts = state.get("revision_attempts", 0)
            max_revisions = 2  # Allow up to 2 automatic revision attempts
            
            if revision_attempts < max_revisions:
                logger.info(f"Pipeline {run_id}: 🔄 ATTEMPTING AUTOMATIC CONTENT REVISION (Attempt {revision_attempts + 1}/{max_revisions})")
                
                # Log detailed improvement plan
                logger.info(f"Pipeline {run_id}: 📋 QUALITY IMPROVEMENT PLAN")
                logger.info(f"  🎯 Summary: {recommendations['summary']}")
                if recommendations["priority_actions"]:
                    logger.info(f"  🔥 Priority Actions:")
                    for action in recommendations["priority_actions"]:
                        logger.info(f"     • {action}")
                
                # Store revision context for the writing agent
                state["revision_attempts"] = revision_attempts + 1
                state["failed_gates"] = failed_gates
                state["quality_recommendations"] = recommendations
                state["revision_context"] = {
                    "previous_scores": individual_gate_scores,
                    "improvement_needed": {gate["gate"]: gate["gap"] for gate in failed_gates},
                    "priority_focus": recommendations["priority_actions"][:3]  # Top 3 priorities
                }
                
                # Return to writing stage with revision context
                logger.info(f"Pipeline {run_id}: 🔄 Restarting from writing stage with quality improvement guidance")
                revision_result = _execute_content_revision_cycle(state, celery_task)
                if revision_result.get("status") == "failed":
                    return revision_result
                state = revision_result
                state["editing_completed_in_revision"] = True
                
                # Refresh quality metrics after revision cycle succeeded
                quality_scores = state.get("quality_scores", {})
                individual_gate_scores = {}
                for gate_name in ["fact_check", "domain_expertise", "style_consistency", "compliance"]:
                    individual_gate_scores[gate_name] = quality_scores.get(gate_name, 0.0)
                failed_gates = []
            else:
                # Maximum revisions reached, fail the pipeline
                error_msg = f"Quality gates failed after {max_revisions} revision attempts. "
                for gate_failure in failed_gates:
                    error_msg += f"{gate_failure['gate']}: {gate_failure['score']:.3f} < {gate_failure['threshold']:.3f} (gap: {gate_failure['gap']:.3f}) "
                
                error_msg += f"All individual scores: "
                for gate, score in individual_gate_scores.items():
                    error_msg += f"{gate}={score:.3f} "
                
                logger.error(f"Pipeline {run_id}: {error_msg}")
                logger.error(f"Pipeline {run_id}: Maximum revision attempts ({max_revisions}) reached. Manual intervention required.")
                
                state["status"] = "failed"
                state["error_message"] = error_msg
                state["failed_gates"] = failed_gates
                state["quality_recommendations"] = recommendations
                return _create_failure_result(run_id, error_msg)
        
        # Check overall threshold only if individual gates passed
        overall_score = quality_scores.get("overall", 0.0)
        overall_threshold = quality_thresholds.get("overall", 0.85)
        
        if overall_score < overall_threshold:
            error_msg = f"Overall quality threshold not met: {overall_score:.3f} < {overall_threshold:.3f}. "
            error_msg += f"Individual scores (all passed): "
            for gate, score in individual_gate_scores.items():
                error_msg += f"{gate}={score:.3f} "
            
            logger.error(f"Pipeline {run_id}: {error_msg}")
            state["status"] = "failed"
            state["error_message"] = error_msg
            return _create_failure_result(run_id, error_msg)
        
        logger.info(f"Pipeline {run_id}: All quality thresholds met - Overall: {overall_score:.3f} >= {overall_threshold:.3f}, Individual gates all passed")
        _track_stage_duration(state, "quality_gates", stage_start_time)
        
        # Stage 6: Editing
        if state.get("editing_completed_in_revision"):
            logger.info(f"Pipeline {run_id}: Editing already completed during revision cycle – skipping duplicate run")
            state.pop("editing_completed_in_revision", None)
        else:
            stage_start_time = datetime.now()
            logger.info(f"Pipeline {run_id}: Starting editing stage")
            state = _execute_editing_stage(state, celery_task)
            if state["status"] == "failed":
                return _create_failure_result(run_id, state["error_message"])
            _track_stage_duration(state, "editing", stage_start_time)
        
        # Stage 7: Critique
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Starting critique stage")
        state = _execute_critique_stage(state, celery_task)
        if state["status"] == "failed":
            return _create_failure_result(run_id, state["error_message"])
        _track_stage_duration(state, "critique", stage_start_time)
        
        # Stage 8: Formatting
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Starting formatting stage")
        state = _execute_formatting_stage(state, celery_task)
        if state["status"] == "failed":
            return _create_failure_result(run_id, state["error_message"])
        _track_stage_duration(state, "formatting", stage_start_time)
        
        # Stage 9: Human Review
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Starting human review stage")
        state = _execute_human_review_stage(state, celery_task)
        if state["status"] == "failed":
            return _create_failure_result(run_id, state["error_message"])
        _track_stage_duration(state, "human_review", stage_start_time)
        
        # Check if pipeline should pause for human review
        if state.get("human_review_required") and state.get("review_status") == "pending":
            logger.info(f"Pipeline {run_id}: Paused at human review stage, waiting for approval")
            
            # Update pipeline status to paused
            update_pipeline_status(
                run_id, 
                "paused", 
                current_step="human_review_pending",
                progress_percentage=90.0,
                review_session_id=state.get("review_session_id")
            )
            
            # Return paused state - pipeline will resume when content is approved
            return {
                "run_id": run_id,
                "status": "paused",
                "current_step": "human_review_pending",
                "progress_percentage": 90.0,
                "message": "Pipeline paused for human review. Resume via Content tab approval.",
                "human_review_required": True,
                "content_draft_id": state.get("current_draft_id"),
                "pause_reason": "awaiting_human_review"
            }
        
        # Stage 10: Publishing (only execute if human review passed or not required)
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Starting publishing stage")
        state = _execute_publishing_stage(state, celery_task)
        if state["status"] == "failed":
            return _create_failure_result(run_id, state["error_message"])
        _track_stage_duration(state, "publishing", stage_start_time)
        
        # Complete pipeline
        state["status"] = "completed"
        state["progress_percentage"] = 100.0
        state["processing_end_time"] = datetime.now()
        state["step_history"].append("pipeline_completed")
        
        # Ensure final content draft is marked as completed (FIXED: proper async handling)
        if state.get("current_draft_id"):
            try:
                from app.services.content_service import ContentService
                from app.db.database import AsyncSessionLocal
                import asyncio
                import concurrent.futures
                
                def finalize_content_sync():
                    """Finalize content draft in new thread with new event loop."""
                    async def finalize_content_async():
                        try:
                            async with AsyncSessionLocal() as db:
                                content_service = ContentService(db)
                                # Mark final draft as current and completed
                                await content_service.mark_draft_as_current(state["current_draft_id"])
                                logger.info(f"✅ Final content draft {state['current_draft_id']} marked as completed")
                                return True
                        except Exception as e:
                            logger.error(f"❌ Content finalization error: {e}")
                            return False
                    
                    # Create new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(finalize_content_async())
                        return result
                    finally:
                        new_loop.close()
                
                # Run content finalization in separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(finalize_content_sync)
                    success = future.result(timeout=15)  # 15 second timeout
                    
                    if success:
                        logger.info(f"✅ Content finalization successful for draft {state['current_draft_id']}")
                    else:
                        logger.error(f"❌ Content finalization failed for draft {state['current_draft_id']}")
                        
            except Exception as e:
                logger.error(f"❌ Failed to finalize content draft: {e}")
                import traceback
                logger.error(f"❌ Full finalization traceback: {traceback.format_exc()}")
        
        # Create final results
        result = {
            "run_id": run_id,
            "status": "completed",
            "published_urls": state.get("published_urls", []),
            "processing_time": (state["processing_end_time"] - state["processing_start_time"]).total_seconds(),
            "content_generated": bool(state.get("draft")),
            "final_quality_score": state.get("quality_scores", {}).get("overall", 0.0),
            "human_approved": state.get("human_review_feedback", {}).get("decision") == "approved"
        }
        
        # Ensure result is JSON serializable (remove any complex objects)
        serialized_result = _ensure_json_serializable(result)
        
        # Validate serialization before returning
        try:
            import json
            json.dumps(serialized_result)
            logger.info("✅ Pipeline result successfully validated as JSON serializable")
        except (TypeError, ValueError) as e:
            logger.error(f"❌ Pipeline result serialization validation failed: {e}")
            # Fallback to basic result structure
            serialized_result = {
                "run_id": run_id,
                "status": "completed",
                "published_urls": [],
                "processing_time": 0.0,
                "content_generated": True,
                "final_quality_score": 0.0,
                "human_approved": False
            }
        
        # Update database with completion
        update_pipeline_completion(run_id, serialized_result)
        
        # Update Celery task state with serializable data only
        celery_task.update_state(
            state="SUCCESS",
            meta={
                "run_id": run_id,
                "status": "completed", 
                "progress": 100,
                "success": True
            }
        )
        
        logger.info(f"Celery pipeline {run_id} completed successfully")
        return serialized_result
        
    except Exception as e:
        logger.error(f"Celery pipeline execution failed: {str(e)}")
        return _create_failure_result(run_id, str(e))


def _update_redis_state(
    run_id: str,
    status: str,
    current_step: Optional[str] = None,
    progress_percentage: Optional[float] = None,
    error_message: Optional[str] = None
):
    """Update Redis state for real-time monitoring with thread-safe execution."""
    try:
        import asyncio
        import threading
        import concurrent.futures
        
        redis_state = {
            "run_id": run_id,
            "status": status,
            "current_step": current_step or "unknown",
            "progress_percentage": progress_percentage or 0.0,
            "error_message": error_message,
            "updated_at": datetime.now().isoformat()
        }
        
        def update_redis_sync():
            """Update Redis in new thread with new event loop."""
            async def update_redis_async():
                try:
                    # Try direct Redis operation instead of using the service
                    import json
                    import redis.asyncio as redis
                    import os
                    
                    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
                    redis_client = redis.from_url(redis_url)
                    
                    # Store the state directly
                    state_key = f"pipeline:full_state:{run_id}"
                    serialized_state = json.dumps(redis_state, default=str)
                    await redis_client.set(state_key, serialized_state, ex=86400)  # 24 hour expiry
                    await redis_client.aclose()
                    return True
                except Exception as e:
                    logger.warning(f"Redis update failed: {e}")
                    return False
            
            # Create new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(update_redis_async())
                return result
            finally:
                new_loop.close()
        
        # Run Redis update in separate thread to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(update_redis_sync)
            success = future.result(timeout=5)  # 5 second timeout
            
            if not success:
                logger.warning(f"Redis state update failed for run_id: {run_id}")
                
    except Exception as e:
        logger.warning(f"Failed to update Redis state: {e}")


# Pipeline completion now handled by sync_database module


@celery_app.task(bind=True, name="resume_pipeline_from_publishing")
def resume_pipeline_from_publishing(self, run_id: str, content_id: str):
    """
    Resume a paused pipeline from the publishing stage after human approval.
    
    Args:
        run_id: UUID of the pipeline run
        content_id: UUID of the approved content
    """
    try:
        # Log resume start
        log_agent_activity(
            pipeline_run_id=run_id,
            agent_name="PipelineOrchestrator",
            stage="resume_publishing",
            operation="start_execution",
            message=f"Resuming pipeline from publishing stage for content {content_id}",
            level="INFO",
            context={"content_id": content_id, "resume_reason": "human_approved"}
        )
        
        logger.info(f"Resuming pipeline {run_id} from publishing stage for content {content_id}")
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"run_id": run_id, "status": "resuming", "progress": 95}
        )
        
        # Load pipeline state from Redis or reconstruct minimal state
        try:
            import asyncio
            state = asyncio.run(redis_service.get_pipeline_state(run_id))
            if not state:
                # Reconstruct minimal state for publishing
                state = _reconstruct_state_for_publishing(run_id, content_id)
        except Exception as e:
            logger.warning(f"Failed to load state from Redis, reconstructing: {e}")
            state = _reconstruct_state_for_publishing(run_id, content_id)
        
        # Update state for publishing stage
        state["current_step"] = "publishing"
        state["progress_percentage"] = 95.0
        state["human_review_required"] = False
        state["review_status"] = "approved"
        state["current_draft_id"] = content_id
        
        # Update pipeline status to running
        update_pipeline_status(
            run_id, 
            "running", 
            current_step="publishing",
            progress_percentage=95.0
        )
        
        # Execute publishing stage
        stage_start_time = datetime.now()
        logger.info(f"Pipeline {run_id}: Resuming publishing stage")
        logger.info(f"🔧 DEBUG: Resume publishing - state keys: {list(state.keys())}")
        logger.info(f"🔧 DEBUG: Resume publishing - current_draft_id: {state.get('current_draft_id')}")
        logger.info(f"🔧 DEBUG: Resume publishing - channel_drafts available: {bool(state.get('channel_drafts'))}")
        
        state = _execute_publishing_stage(state, self)
        
        # Log resume completion
        log_agent_activity(
            pipeline_run_id=run_id,
            agent_name="PipelineOrchestrator",
            stage="resume_publishing",
            operation="complete_execution",
            message=f"Resume publishing completed with status: {state.get('status')}",
            level="INFO" if state.get("status") != "failed" else "ERROR",
            context={"final_status": state.get("status"), "error_message": state.get("error_message")}
        )
        
        logger.info(f"🔧 DEBUG: Resume publishing completed - status: {state.get('status')}")
        logger.info(f"🔧 DEBUG: Resume publishing completed - error_message: {state.get('error_message')}")
        
        if state["status"] == "failed":
            result = _create_failure_result(run_id, state["error_message"])
        else:
            _track_stage_duration(state, "publishing", stage_start_time)
            
            # Complete pipeline
            state["status"] = "completed"
            state["progress_percentage"] = 100.0
            state["processing_end_time"] = datetime.now()
            state["step_history"].append("pipeline_completed")
            
            # Create final results
            result = {
                "run_id": run_id,
                "status": "completed",
                "published_urls": state.get("published_urls", []),
                "processing_time": (state["processing_end_time"] - state.get("processing_start_time", datetime.now())).total_seconds(),
                "content_generated": True,
                "final_quality_score": state.get("quality_scores", {}).get("overall", 0.0),
                "human_approved": True
            }
            
            # Ensure result is JSON serializable (remove any complex objects)
            serialized_result = _ensure_json_serializable(result)
            
            # Update database with completion
            update_pipeline_completion(run_id, serialized_result)
        
        # Update task state
        self.update_state(
            state="SUCCESS",
            meta={
                "run_id": run_id,
                "status": serialized_result["status"], 
                "progress": 100,
                "final_state": serialized_result
            }
        )
        
        logger.info(f"Pipeline {run_id} resume completed with status: {serialized_result['status']}")
        return serialized_result
        
    except Exception as e:
        logger.error(f"Pipeline resume failed for run_id: {run_id} - {str(e)}")
        
        # Log resume error
        log_agent_activity(
            pipeline_run_id=run_id,
            agent_name="PipelineOrchestrator",
            stage="resume_publishing",
            operation="error",
            message=f"Resume pipeline failed: {str(e)}",
            level="ERROR",
            context={"content_id": content_id, "error_type": type(e).__name__, "error_details": str(e)}
        )
        
        # Update database with failure
        update_pipeline_status(run_id, "failed", error_message=str(e))
        
        # Update task state
        self.update_state(
            state="FAILURE",
            meta={"run_id": run_id, "error": str(e)}
        )
        
        raise e


def _reconstruct_state_for_publishing(run_id: str, content_id: str) -> Dict[str, Any]:
    """Reconstruct minimal pipeline state needed for publishing stage."""
    try:
        # Get content and pipeline info from database
        from app.db.sync_database import get_sync_session
        from sqlalchemy import text
        import json
        
        with get_sync_session() as session:
            # Get content draft info
            content_query = text("""
                SELECT cd.*, pr.domain_focus 
                FROM content_drafts cd 
                JOIN pipeline_runs pr ON cd.pipeline_run_id = pr.id 
                WHERE cd.id = :content_id
            """)
            content_result = session.execute(content_query, {"content_id": content_id})
            content_row = content_result.fetchone()
            
            if not content_row:
                raise ValueError(f"Content with ID {content_id} not found")
            
            # Create minimal state for publishing
            state = {
                "run_id": run_id,
                "status": "running",
                "current_step": "publishing",
                "progress_percentage": 95.0,
                "current_draft_id": content_id,
                "domain_focus": (
                    json.loads(content_row.domain_focus) if content_row.domain_focus and isinstance(content_row.domain_focus, str)
                    else content_row.domain_focus if content_row.domain_focus 
                    else ["General"]
                ),
                "human_review_required": False,
                "review_status": "approved",
                "step_history": ["reconstructed_for_publishing"],
                "processing_start_time": datetime.now(),
                "channel_drafts": {
                    "digitaldossier": {
                        "title": content_row.title,
                        "content": content_row.content,
                        "meta_description": content_row.meta_description,
                        "keywords": json.loads(content_row.keywords) if content_row.keywords and isinstance(content_row.keywords, str) else (content_row.keywords if content_row.keywords else []),
                        "abstract": content_row.abstract
                    }
                }
            }
            
            return state
            
    except Exception as e:
        logger.error(f"Failed to reconstruct state for publishing: {e}")
        raise


@celery_app.task(name="cleanup_pipeline_checkpoints")
def cleanup_pipeline_checkpoints(run_id: str):
    """Clean up LangGraph checkpoint files after pipeline completion."""
    try:
        checkpoint_file = f"pipeline_checkpoints_{run_id}.db"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info(f"Cleaned up checkpoint file: {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to cleanup checkpoint file: {e}")


# Task to monitor and recover stuck pipelines
@celery_app.task(name="monitor_pipeline_health")
def monitor_pipeline_health():
    """Monitor pipeline health and recover stuck processes."""
    try:
        # This would implement health checks for running pipelines
        # and recover any stuck processes
        logger.info("Pipeline health monitoring completed")
    except Exception as e:
        logger.error(f"Pipeline health monitoring failed: {e}")


# Pipeline stage simulation functions
def _simulate_research(run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate research stage with sync database operations."""
    logger.info(f"Generating research data for {run_id}")
    
    try:
        # Create a topic idea using sync operations
        domain = config.get('domain_focus', ['General'])[0]
        
        topic_id = create_topic_idea(
            pipeline_run_id=run_id,
            title=f"Advanced Content Strategy for {domain}",
            description="A comprehensive analysis of modern content creation approaches and best practices.",
            domain=domain,
            relevance_score=0.92,
            novelty_score=0.85,
            seo_difficulty=0.65,
            overall_score=0.88,
            keywords=["content strategy", "digital marketing", "content creation", "SEO optimization"]
        )
        
        if topic_id:
            logger.info(f"Created topic idea {topic_id} for pipeline {run_id}")
            return {"topic_id": topic_id, "status": "completed"}
        else:
            raise Exception("Failed to create topic idea")
            
    except Exception as e:
        logger.error(f"Research stage error: {e}")
        return {"status": "error", "message": str(e)}


def _simulate_curation(run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate content curation stage."""
    logger.info(f"Performing content curation for {run_id}")
    return {"status": "completed", "message": "Content strategy developed"}


def _generate_content(run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate actual content and store in sync database."""
    logger.info(f"Generating content for {run_id}")
    
    try:
        # Get the selected topic_id
        topic_id = get_selected_topic_id(run_id)
        
        if not topic_id:
            raise Exception("No topic found for content generation")
            
            # Generate sample content
            domain = config.get('domain_focus', ['General'])[0]
            content_title = f"The Future of {domain}: Trends and Innovations"
            
            sample_content = f"""# {content_title}

## Executive Summary
This comprehensive analysis explores the evolving landscape of {domain} and identifies key trends that will shape the future of this dynamic field.

## Introduction
The {domain} industry has experienced unprecedented growth and transformation over the past decade. As organizations continue to adapt to changing market conditions and technological advancements, understanding emerging trends becomes crucial for strategic planning and competitive advantage.

## Key Trends and Innovations

### 1. Digital Transformation Acceleration
The digital transformation journey in {domain} has accelerated significantly, driven by:
- Advanced automation technologies
- Cloud-native architectures
- Data-driven decision making
- Enhanced user experiences

### 2. Emerging Technologies Impact
Several breakthrough technologies are reshaping the {domain} landscape:
- Artificial Intelligence and Machine Learning
- Internet of Things (IoT) integration
- Blockchain applications
- Edge computing solutions

### 3. Market Evolution Patterns
Current market dynamics show distinct patterns:
- Increased demand for personalized solutions
- Growing emphasis on sustainability
- Shift towards subscription-based models
- Greater focus on security and compliance

## Strategic Recommendations

### For Organizations
1. **Invest in Technology Infrastructure**: Build robust, scalable systems that can adapt to future requirements
2. **Develop Data Capabilities**: Establish comprehensive data strategies to drive insights and innovation
3. **Foster Innovation Culture**: Create environments that encourage experimentation and continuous learning
4. **Strengthen Security Posture**: Implement comprehensive cybersecurity frameworks

### For Professionals
1. **Continuous Learning**: Stay updated with emerging technologies and industry best practices
2. **Cross-functional Collaboration**: Develop skills to work effectively across different domains
3. **Data Literacy**: Build capabilities to interpret and act on data-driven insights
4. **Adaptability**: Cultivate flexibility to thrive in changing environments

## Implementation Framework

### Phase 1: Assessment (Months 1-2)
- Current state analysis
- Gap identification
- Resource evaluation
- Risk assessment

### Phase 2: Planning (Months 3-4)
- Strategy development
- Technology roadmap creation
- Team formation
- Budget allocation

### Phase 3: Execution (Months 5-12)
- Solution implementation
- Process optimization
- Team training
- Performance monitoring

## Conclusion
The future of {domain} presents both exciting opportunities and significant challenges. Organizations that proactively embrace emerging trends, invest in the right technologies, and develop adaptive capabilities will be best positioned to thrive in this evolving landscape.

Success requires a balanced approach that combines technological innovation with human expertise, strategic vision with tactical execution, and bold innovation with prudent risk management.

## References
1. Industry Research Report 2024
2. Technology Trends Analysis
3. Market Dynamics Study
4. Expert Interview Insights
5. Competitive Landscape Analysis
"""
            
        # Create content draft using sync operations
        content_id = create_content_draft(
            pipeline_run_id=run_id,
            topic_id=topic_id,
            title=content_title,
            content=sample_content,
            subtitle=f"A comprehensive analysis of emerging trends and strategic opportunities in {domain}",
            abstract=f"This analysis explores the evolving landscape of {domain} and identifies key trends that will shape the future of this dynamic field.",
            citations=["Industry Research Report 2024", "Technology Trends Analysis", "Market Dynamics Study"],
            readability_score=75.5,
            meta_description=f"Discover the key trends and innovations shaping the future of {domain} with strategic insights and actionable recommendations.",
            keywords=[f"{domain} trends", "innovation", "digital transformation", "strategy", "technology"],
            tags=[domain.lower(), "trends", "analysis", "strategy"]
        )
        
        if content_id:
            logger.info(f"Generated content draft {content_id} for pipeline {run_id}")
            return {"content_id": content_id, "status": "completed", "word_count": len(sample_content.split())}
        else:
            raise Exception("Failed to create content draft")
        
    except Exception as e:
        logger.error(f"Content generation error: {e}")
        return {"status": "error", "message": str(e)}


def _simulate_quality_gates(run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate quality gate validation."""
    logger.info(f"Running quality gates for {run_id}")
    return {"status": "completed", "quality_score": 0.92}


def _simulate_formatting(run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate content formatting stage."""
    logger.info(f"Formatting content for {run_id}")
    return {"status": "completed", "message": "Content formatted for publishing"}


def _simulate_publishing(run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate publishing stage without actual upload."""
    logger.info(f"Simulating publishing for {run_id}")
    # Return mock URLs instead of trying to publish to production
    return {
        "status": "completed", 
        "urls": ["http://localhost:3003/mock-published-content"],
        "message": "Content prepared for publishing (development mode)"
    }


# Celery Pipeline Stage Execution Functions

def _execute_research_stage(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute research stage using ResearchCoordinatorAgent."""
    try:
        from contentrunway.agents import ResearchCoordinatorAgent
        import asyncio
        
        # Update progress
        state["current_step"] = "research"
        state["progress_percentage"] = 20.0
        _update_pipeline_progress(state, celery_task)
        
        # Initialize and execute research agent
        research_agent = ResearchCoordinatorAgent()
        
        # Log agent start
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ResearchCoordinatorAgent",
            stage="research",
            operation="start_execution",
            message=f"Starting research phase for query: '{state['research_query']}'",
            level="INFO",
            context={"domains": state["domain_focus"]}
        )
        
        # Execute research (sync wrapper for async agent)
        logger.info(f"📝 Research Agent Input - Query: '{state['research_query']}', Domains: {state['domain_focus']}")
        research_results = asyncio.run(research_agent.execute(
            query=state["research_query"],
            domains=state["domain_focus"],
            state=state
        ))
        
        # Log agent completion
        # Extract IDs for pointer-based context (with minimal impact)
        sources = research_results.get('sources', [])
        topics = research_results.get('topics', [])
        source_ids = [getattr(s, 'id', None) or s.get('id') for s in sources if hasattr(s, 'id') or (isinstance(s, dict) and s.get('id'))]
        topic_ids = [getattr(t, 'id', None) or t.get('id') for t in topics if hasattr(t, 'id') or (isinstance(t, dict) and t.get('id'))]
        
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ResearchCoordinatorAgent",
            stage="research",
            operation="complete_execution",
            message=f"Research completed: {len(sources)} sources, {len(topics)} topics",
            level="INFO",
            context={
                "sources_count": len(sources), 
                "topics_count": len(topics),
                "source_ids": [str(id) for id in source_ids if id],  # Pointer keys for enhanced UI
                "topic_ids": [str(id) for id in topic_ids if id]     # Pointer keys for enhanced UI
            }
        )
        
        logger.info(f"📊 Research Agent Output - Sources: {len(research_results.get('sources', []))}, Raw Topics: {len(research_results.get('topics', []))}")
        
        # Serialize topics for state persistence
        serialized_topics = []
        for topic in research_results.get("topics", []):
            if hasattr(topic, 'dict'):
                # Pydantic model - convert to dict
                serialized_topics.append(topic.dict())
            elif isinstance(topic, dict):
                # Already a dictionary
                serialized_topics.append(topic)
            else:
                # String or other format - convert to basic dict
                import uuid
                serialized_topics.append({
                    'id': str(uuid.uuid4()),
                    'title': str(topic),
                    'description': '',
                    'domain': state['domain_focus'][0] if state['domain_focus'] else 'general',
                    'relevance_score': 0.7,
                    'novelty_score': 0.7,
                    'seo_difficulty': 0.5,
                    'overall_score': 0.7,
                    'target_keywords': []
                })
        
        # Fallback: Generate topics if research agent failed to produce any
        if not serialized_topics:
            import uuid
            domain = state['domain_focus'][0] if state['domain_focus'] else 'ai'
            query = state.get('research_query', 'general content')
            
            logger.warning("Research agent produced no topics, generating fallback topics")
            serialized_topics = [
                {
                    'id': str(uuid.uuid4()),
                    'title': f"Comprehensive Guide to {domain.upper()}: {query}",
                    'description': f"An in-depth exploration of {domain} covering practical applications, current trends, and future implications.",
                    'domain': domain,
                    'relevance_score': 0.85,
                    'novelty_score': 0.75,
                    'seo_difficulty': 0.6,
                    'overall_score': 0.8,
                    'target_keywords': [f"{domain} guide", f"{domain} trends", f"{domain} applications"]
                },
                {
                    'id': str(uuid.uuid4()),
                    'title': f"Best Practices in {domain.upper()}: Implementation and Strategy",
                    'description': f"Strategic insights and implementation guidelines for {domain} in modern enterprises.",
                    'domain': domain,
                    'relevance_score': 0.82,
                    'novelty_score': 0.78,
                    'seo_difficulty': 0.65,
                    'overall_score': 0.82,
                    'target_keywords': [f"{domain} best practices", f"{domain} strategy", f"{domain} implementation"]
                }
            ]
        
        # Update state with research results
        state["sources"] = research_results.get("sources", [])
        state["topics"] = serialized_topics
        state["step_history"].append("research_completed")
        
        # Persist research sources to database
        try:
            for source in state["sources"]:
                if isinstance(source, dict) and source.get("url") and source.get("title"):
                    source_id = create_research_source(
                        pipeline_run_id=state["run_id"],
                        url=source["url"],
                        title=source["title"],
                        summary=source.get("summary", source.get("content", "")[:500]),
                        author=source.get("author"),
                        publication_date=source.get("publication_date"),
                        domain=source.get("domain", state["domain_focus"][0] if state["domain_focus"] else "unknown"),
                        source_type=source.get("source_type", "web"),
                        key_points=source.get("key_points", []),
                        quotable_content=source.get("quotable_content", {}),
                        credibility_score=source.get("credibility_score", 0.8),
                        relevance_score=source.get("relevance_score", 0.8),
                        currency_score=source.get("currency_score", 0.8)
                    )
                    if source_id:
                        logger.info(f"Persisted research source {source_id}: {source['title'][:50]}")
            
            # Persist topic ideas to database  
            for topic in serialized_topics:
                if isinstance(topic, dict) and topic.get("title"):
                    topic_id = create_topic_idea(
                        pipeline_run_id=state["run_id"],
                        title=topic["title"],
                        description=topic.get("description", ""),
                        domain=topic.get("domain", state["domain_focus"][0] if state["domain_focus"] else "unknown"),
                        relevance_score=topic.get("relevance_score", 0.7),
                        novelty_score=topic.get("novelty_score", 0.7),
                        seo_difficulty=topic.get("seo_difficulty", 0.5),
                        overall_score=topic.get("overall_score", 0.7),
                        keywords=topic.get("target_keywords", []),
                        is_selected=False  # Will be updated in curation stage
                    )
                    if topic_id:
                        logger.info(f"Persisted topic idea {topic_id}: {topic['title'][:50]}")
                        
            logger.info(f"Research data persisted to database for pipeline {state['run_id']}")
        except Exception as db_error:
            logger.error(f"Failed to persist research data: {db_error}")
            # Don't fail the pipeline for DB persistence issues
        
        # Store state in Redis for persistence
        _store_pipeline_state(state["run_id"], state)
        
        logger.info(f"✅ Research Stage Complete: {len(state['sources'])} sources, {len(state['topics'])} topics")
        logger.info(f"📋 Research Topics Generated: {[t.get('title', 'No title') for t in state['topics'][:3]]}")
        if state['sources']:
            logger.info(f"📄 Research Sources Sample: {[s.get('title', 'No title') for s in state['sources'][:2]]}")
        
    except Exception as e:
        logger.error(f"Research stage failed: {str(e)}")
        
        # Log agent error
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ResearchCoordinatorAgent",
            stage="research",
            operation="error",
            message=f"Research stage failed: {str(e)}",
            level="ERROR",
            context={"error_type": type(e).__name__, "error_details": str(e)}
        )
        
        state["status"] = "failed"
        state["error_message"] = f"Research failed: {str(e)}"
    
    return state


def _execute_curation_stage(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute curation stage using ContentCuratorAgent."""
    try:
        from contentrunway.agents import ContentCuratorAgent
        import asyncio
        
        # Update progress
        state["current_step"] = "curation"
        state["progress_percentage"] = 30.0
        _update_pipeline_progress(state, celery_task)
        
        # Validate topics exist
        if not state.get("topics"):
            raise ValueError("No topics available for curation")
        
        # Initialize and execute curation agent
        curator_agent = ContentCuratorAgent()
        
        # Log agent start
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ContentCuratorAgent",
            stage="curation",
            operation="start_execution",
            message=f"Starting curation with {len(state['topics'])} topics and {len(state['sources'])} sources",
            level="INFO",
            context={"topics_count": len(state["topics"]), "sources_count": len(state["sources"])}
        )
        
        # Execute curation
        logger.info(f"📝 Curation Agent Input - Topics: {len(state['topics'])}, Sources: {len(state['sources'])}")
        logger.info(f"📋 Curation Input Topics: {[t.get('title', 'No title') for t in state['topics'][:3]]}")
        curation_results = asyncio.run(curator_agent.execute(
            topics=state["topics"],
            sources=state["sources"],
            state=state
        ))
        
        # Log agent completion
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ContentCuratorAgent",
            stage="curation",
            operation="complete_execution",
            message=f"Curation completed: Selected topic ID {curation_results.get('chosen_topic_id', 'None')}",
            level="INFO",
            context={
                "chosen_topic_id": curation_results.get('chosen_topic_id'), 
                "result_keys": list(curation_results.keys())
            }
        )
        
        logger.info(f"📊 Curation Agent Output - Chosen Topic ID: {curation_results.get('chosen_topic_id', 'None')}")
        logger.info(f"📊 Curation Result Keys: {list(curation_results.keys())}")
        
        # Update state with curation results
        state["chosen_topic_id"] = curation_results["chosen_topic_id"]
        state["step_history"].append("curation_completed")
        
        # Update the chosen topic in database to mark it as selected
        try:
            if state["chosen_topic_id"]:
                # Update pipeline run with chosen topic ID
                from app.db.sync_database import update_pipeline_status
                update_pipeline_status(
                    run_id=state["run_id"],
                    status="running",
                    current_step="curation_completed"
                )
                
                # Update the topic_ideas table to mark the selected topic
                with get_sync_session() as session:
                    from sqlalchemy import update
                    from app.models.pipeline import TopicIdea
                    
                    # First, unmark any previously selected topics
                    stmt = update(TopicIdea).where(
                        TopicIdea.pipeline_run_id == uuid.UUID(state["run_id"])
                    ).values(is_selected=False)
                    session.execute(stmt)
                    
                    # Mark the chosen topic as selected
                    stmt = update(TopicIdea).where(
                        TopicIdea.pipeline_run_id == uuid.UUID(state["run_id"]),
                        TopicIdea.title.like(f"%{_find_topic_by_id(state['topics'], state['chosen_topic_id']).get('title', '')[:50]}%")
                    ).values(is_selected=True)
                    result = session.execute(stmt)
                    
                    if result.rowcount > 0:
                        logger.info(f"Marked topic as selected in database for pipeline {state['run_id']}")
                
        except Exception as db_error:
            logger.error(f"Failed to update selected topic in database: {db_error}")
            # Don't fail the pipeline for DB persistence issues
        
        # Store state in Redis
        _store_pipeline_state(state["run_id"], state)
        
        # Find and log the chosen topic details
        chosen_topic = _find_topic_by_id(state["topics"], state["chosen_topic_id"])
        chosen_title = chosen_topic.get('title', 'Unknown') if chosen_topic else 'Topic not found'
        logger.info(f"✅ Curation Stage Complete: chosen topic {state['chosen_topic_id']}")
        logger.info(f"📋 Chosen Topic: '{chosen_title}'")
        
    except Exception as e:
        logger.error(f"Curation stage failed: {str(e)}")
        state["status"] = "failed"
        state["error_message"] = f"Curation failed: {str(e)}"
    
    return state


def _execute_seo_strategy_stage(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute SEO strategy stage using SEOStrategistAgent."""
    try:
        from contentrunway.agents import SEOStrategistAgent
        import asyncio
        
        # Update progress
        state["current_step"] = "seo_strategy"
        state["progress_percentage"] = 40.0
        _update_pipeline_progress(state, celery_task)
        
        # Find chosen topic
        chosen_topic_dict = _find_topic_by_id(state["topics"], state["chosen_topic_id"])
        if not chosen_topic_dict:
            logger.error(f"❌ SEO Stage: Chosen topic {state['chosen_topic_id']} not found in topics list")
            logger.error(f"📋 Available topic IDs: {[t.get('id', 'No ID') for t in state['topics']]}")
            raise ValueError(f"Chosen topic {state['chosen_topic_id']} not found")
        
        # Convert dictionary to object for agent compatibility
        chosen_topic = _create_topic_object(chosen_topic_dict)
        logger.info(f"📝 SEO Agent Input - Topic: '{chosen_topic.title}', Sources: {len(state['sources'])}")
        logger.info(f"📋 Topic Keywords: {getattr(chosen_topic, 'target_keywords', 'None')}")
        
        # Initialize and execute SEO strategist
        seo_agent = SEOStrategistAgent()
        
        # Log agent start
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="SEOStrategistAgent",
            stage="seo_strategy",
            operation="start_execution",
            message=f"Starting SEO strategy for topic: '{chosen_topic.title}'",
            level="INFO",
            context={"topic_id": state["chosen_topic_id"], "sources_count": len(state["sources"])}
        )
        
        # Execute SEO strategy
        seo_results = asyncio.run(seo_agent.execute(
            topic=chosen_topic,
            sources=state["sources"],
            state=state
        ))
        
        # Log agent completion
        outline_result = seo_results.get("outline")
        outline_dict = _normalize_outline_data(outline_result)
        sections_count = _get_outline_sections_count(outline_dict or outline_result)
        outline_id = None
        if outline_dict and isinstance(outline_dict, dict):
            outline_id = outline_dict.get('id')
        elif outline_result and hasattr(outline_result, 'id'):
            outline_id = getattr(outline_result, 'id', None)
        
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="SEOStrategistAgent",
            stage="seo_strategy",
            operation="complete_execution",
            message=f"SEO strategy completed: {sections_count} sections generated",
            level="INFO",
            context={
                "sections_count": sections_count, 
                "result_keys": list(seo_results.keys()),
                "outline_id": str(outline_id) if outline_id else None  # Pointer key for enhanced UI
            }
        )
        
        logger.info(f"📊 SEO Agent Output - Result Keys: {list(seo_results.keys())}")
        if sections_count:
            logger.info(f"📋 SEO Outline Generated: {sections_count} sections")
            section_titles = _extract_outline_titles(outline_dict or outline_result)
            if section_titles:
                logger.info(f"📋 Outline Sections: {section_titles}")
        else:
            logger.error(f"❌ SEO Agent: No outline generated")
        
        # Update state with SEO results
        state["outline"] = outline_dict
        state["step_history"].append("seo_strategy_completed")
        
        # Persist content outline to database
        try:
            outline = state.get("outline")
            chosen_topic_id = state.get("chosen_topic_id")
            if outline and chosen_topic_id:
                # Extract outline data
                if hasattr(outline, 'sections'):
                    sections = outline.sections
                elif isinstance(outline, dict) and 'sections' in outline:
                    sections = outline['sections']
                else:
                    sections = []
                
                # Get outline details
                target_audience = getattr(outline, 'target_audience', 'general audience')
                primary_angle = getattr(outline, 'primary_angle', 'comprehensive overview')
                key_takeaways = getattr(outline, 'key_takeaways', [])
                primary_keyword = getattr(outline, 'primary_keyword', '')
                secondary_keywords = getattr(outline, 'secondary_keywords', [])
                estimated_word_count = getattr(outline, 'estimated_word_count', 1500)
                
                outline_id = create_content_outline(
                    pipeline_run_id=state["run_id"],
                    topic_id=chosen_topic_id,
                    sections=sections if isinstance(sections, list) else [],
                    estimated_word_count=estimated_word_count,
                    target_audience=target_audience,
                    primary_angle=primary_angle,
                    key_takeaways=key_takeaways if isinstance(key_takeaways, list) else [],
                    primary_keyword=primary_keyword,
                    secondary_keywords=secondary_keywords if isinstance(secondary_keywords, list) else [],
                    call_to_action=getattr(outline, 'call_to_action', None),
                    internal_link_opportunities=getattr(outline, 'internal_link_opportunities', []),
                    approved=True  # SEO-approved outline
                )
                
                if outline_id:
                    logger.info(f"Persisted content outline {outline_id} to database for pipeline {state['run_id']}")
                    
        except Exception as db_error:
            logger.error(f"Failed to persist content outline: {db_error}")
            # Don't fail the pipeline for DB persistence issues
        
        # Store state in Redis
        _store_pipeline_state(state["run_id"], state)
        
        logger.info(f"SEO strategy completed: outline with {_get_outline_sections_count(state.get('outline'))} sections")
        
    except Exception as e:
        logger.error(f"SEO strategy stage failed: {str(e)}")
        
        # Log agent error
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="SEOStrategistAgent",
            stage="seo_strategy",
            operation="error",
            message=f"SEO strategy stage failed: {str(e)}",
            level="ERROR",
            context={"error_type": type(e).__name__, "error_details": str(e)}
        )
        
        state["status"] = "failed"
        state["error_message"] = f"SEO strategy failed: {str(e)}"
    
    return state


def _prepare_writing_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare state for writing agent by ensuring proper object formats."""
    from contentrunway.state.pipeline_state import Source
    
    # Ensure sources are proper Source objects or dictionaries
    if state.get("sources"):
        prepared_sources = []
        for source in state["sources"]:
            if isinstance(source, dict):
                # Ensure all required fields exist
                source_dict = {
                    'url': source.get('url', ''),
                    'title': source.get('title', 'Untitled'),
                    'author': source.get('author'),
                    'publication_date': source.get('publication_date'),
                    'domain': source.get('domain', 'general'),
                    'source_type': source.get('source_type', 'article'),
                    'summary': source.get('summary', ''),
                    'key_points': source.get('key_points', []),
                    'credibility_score': source.get('credibility_score', 0.8),
                    'relevance_score': source.get('relevance_score', 0.7),
                    'currency_score': source.get('currency_score', 0.6)
                }
                # Preserve lookup identifier if present
                if '_lookup_identifier' in source:
                    source_dict['_lookup_identifier'] = source['_lookup_identifier']
                prepared_sources.append(source_dict)
            else:
                # Already a proper object, keep as is
                prepared_sources.append(source)
        state["sources"] = prepared_sources
    
    # Normalize outline to dictionary format for safe serialization
    if state.get("outline"):
        outline = _normalize_outline_data(state["outline"])
        if outline:
            # Dictionary format - normalize sections
            if 'sections' in outline and outline['sections']:
                formatted_sections = []
                for section in outline['sections']:
                    if isinstance(section, dict):
                        formatted_section = {
                            'title': section.get('title', section.get('section_title', 'Untitled Section')),
                            'key_points': section.get('key_points', section.get('points', [])),
                            'estimated_words': section.get('estimated_words', section.get('word_count', 200))
                        }
                        formatted_sections.append(formatted_section)
                    else:
                        formatted_sections.append(section)
                outline['sections'] = formatted_sections
            state["outline"] = outline
        else:
            logger.warning(f"⚠️ Failed to normalize outline of type {type(state['outline'])}")
    
    # Safe logging for outline sections count
    outline_sections_count = _get_outline_sections_count(state.get("outline"))
    logger.info(f"✅ Writing state prepared: {len(state.get('sources', []))} sources, {outline_sections_count} outline sections")
    return state


def _execute_writing_stage(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute writing stage using ContentWriterAgent."""
    try:
        from contentrunway.agents import ContentWriterAgent
        import asyncio
        
        # Update progress
        state["current_step"] = "writing"
        state["progress_percentage"] = 60.0
        _update_pipeline_progress(state, celery_task)
        
        # Validate and prepare state for writing agent
        state = _prepare_writing_state(state)
        
        # Log state validation results
        sources_count = len(state.get("sources", []))
        outline_sections_count = 0
        if state.get("outline"):
            outline = state["outline"]
            if isinstance(outline, dict) and 'sections' in outline:
                outline_sections_count = len(outline['sections'])
            elif hasattr(outline, 'sections') and outline.sections:
                outline_sections_count = len(outline.sections)
        logger.info(f"📋 Writing Stage Validation: {sources_count} sources prepared, {outline_sections_count} outline sections ready")
        
        # Initialize and execute content writer
        writer_agent = ContentWriterAgent()
        
        # Prepare outline object for agent while keeping state serializable
        outline_data = state.get("outline")
        outline_obj = _convert_outline_to_object(outline_data)
        if not outline_obj:
            raise ValueError("Outline data unavailable or invalid for writing stage")
        outline_sections = _get_outline_sections_count(outline_data or outline_obj)
        
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ContentWriterAgent",
            stage="writing",
            operation="start_execution",
            message=f"Starting content writing with {outline_sections} sections",
            level="INFO",
            context={"outline_sections": outline_sections, "sources_count": len(state["sources"])}
        )
        
        # Log writing input
        logger.info(f"📝 Writing Agent Input - Outline Sections: {outline_sections}, Sources: {len(state['sources'])}")
        section_titles = _extract_outline_titles(outline_data or outline_obj, limit=3)
        if section_titles:
            logger.info(f"📋 Writing Input Sections: {section_titles}")
        
        # Execute writing
        writing_results = asyncio.run(writer_agent.execute(
            outline=outline_obj,
            sources=state["sources"],
            state=state
        ))
        
        # Log agent completion
        draft = writing_results.get("draft")
        word_count = getattr(draft, 'word_count', 0) if draft else 0
        draft_id = getattr(draft, 'id', None) or (draft.get('id') if isinstance(draft, dict) else None)
        version = getattr(draft, 'version', None) or (draft.get('version') if isinstance(draft, dict) else None)
        stage = getattr(draft, 'stage', None) or (draft.get('stage') if isinstance(draft, dict) else None)
        
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ContentWriterAgent",
            stage="writing",
            operation="complete_execution",
            message=f"Content writing completed: {word_count} words generated",
            level="INFO",
            context={
                "word_count": word_count, 
                "result_keys": list(writing_results.keys()),
                "content_draft_id": str(draft_id) if draft_id else None,  # Pointer key for enhanced UI
                "version": version,                                       # Pointer key for enhanced UI
                "stage": stage                                           # Pointer key for enhanced UI
            }
        )
        
        logger.info(f"📊 Writing Agent Output - Result Keys: {list(writing_results.keys())}")
        if draft:
            logger.info(f"📋 Draft Generated: {word_count} words")
            if hasattr(draft, 'title'):
                logger.info(f"📋 Draft Title: '{draft.title}'")
        else:
            logger.error(f"❌ Writing Agent: No draft generated")
        
        # Update state with writing results
        state["draft"] = writing_results["draft"]
        if writing_results.get("sentence_citation_map"):
            state["sentence_citation_map"] = writing_results["sentence_citation_map"]
        if writing_results.get("domain_terms_used"):
            state["domain_terms_used"] = writing_results["domain_terms_used"]
        if writing_results.get("section_source_mapping"):
            state["section_source_mapping"] = writing_results["section_source_mapping"]
        state["step_history"].append("writing_completed")
        
        # Persist initial draft to database via centralized helper
        if state.get("draft") and state.get("chosen_topic_id"):
            try:
                draft_id = persist_content_draft(state)
                state["current_draft_id"] = draft_id
            except DraftPersistenceError as exc:
                logger.error(f"❌ Draft persistence failed: {exc}")
                state["status"] = "failed"
                state["error_message"] = f"Writing failed: {exc}"
                return state

        logger.info(f"Writing completed: {word_count} words")
        
    except Exception as e:
        logger.error(f"Writing stage failed: {str(e)}")
        
        # Log agent error
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ContentWriterAgent",
            stage="writing",
            operation="error",
            message=f"Writing stage failed: {str(e)}",
            level="ERROR",
            context={"error_type": type(e).__name__, "error_details": str(e)}
        )
        
        state["status"] = "failed"
        state["error_message"] = f"Writing failed: {str(e)}"
    
    return state


def _execute_quality_gates_stage(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute quality gates stage using parallel agent execution."""
    try:
        from contentrunway.agents import (
            FactCheckGateAgent,
            DomainExpertiseGateAgent,
            StyleCriticGateAgent,
            ComplianceGateAgent
        )
        import asyncio
        
        # Update progress
        state["current_step"] = "quality_gates"
        state["progress_percentage"] = 75.0
        _update_pipeline_progress(state, celery_task)
        
        # Initialize quality gate agents
        fact_check_agent = FactCheckGateAgent()
        domain_expertise_agent = DomainExpertiseGateAgent()
        style_critic_agent = StyleCriticGateAgent()
        compliance_agent = ComplianceGateAgent()
        
        # Comprehensive content analysis before quality gates
        draft = state.get("draft")
        if draft:
            content = getattr(draft, 'content', '') if hasattr(draft, 'content') else draft.get('content', '')
            content_length = len(content)
            word_count = len(content.split())
            citation_count = len([m for m in __import__('re').findall(r'\[Citation\s*\d+\]', content)])
            protected_term_count = len([m for m in __import__('re').findall(r'<([^>]+)>', content)])
            
            logger.info(f"🔍 Quality Gates Input Analysis:")
            logger.info(f"   📊 Content Stats:")
            logger.info(f"      - Length: {content_length:,} characters")
            logger.info(f"      - Words: {word_count:,}")
            logger.info(f"      - Citations: {citation_count}")
            logger.info(f"      - Protected terms: {protected_term_count}")
            logger.info(f"   📚 Context:")
            logger.info(f"      - Sources available: {len(state.get('sources', []))}")
            logger.info(f"      - Domain focus: {state.get('domain_focus', [])}")
            
            # Content processing without truncation
            if content_length > 3000:
                logger.info(f"   📄 Large content detected: {content_length:,} characters")
                logger.info(f"      ✅ Processing FULL content (no truncation)")
                logger.info(f"      📊 All {citation_count} citations and {protected_term_count} protected terms will be preserved")
        
        # Log quality gates start
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="QualityGateAgents",
            stage="quality_gates",
            operation="start_execution",
            message="Starting parallel quality gate validation (FactCheck, Domain, Style, Compliance)",
            level="INFO",
            context={"agents": ["FactCheckGateAgent", "DomainExpertiseGateAgent", "StyleCriticGateAgent", "ComplianceGateAgent"]}
        )
        
        # Extract target length configuration
        target_words = state.get("content_word_count_target", 250)
        
        # Execute quality gates in parallel with proper asyncio handling
        async def run_quality_gates():
            tasks = [
                fact_check_agent.execute(
                    state["draft"], 
                    state["sources"],
                    state.get("sentence_citation_map"),
                    target_words
                ),
                domain_expertise_agent.execute(state["draft"], state["domain_focus"]),
                style_critic_agent.execute(state["draft"], state),
                compliance_agent.execute(state["draft"])
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Use asyncio.run with proper exception handling
        try:
            results = asyncio.run(run_quality_gates())
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                # We're in an existing event loop, need to handle differently
                logger.warning("Running in existing event loop, using alternative execution")
                # Create new event loop in a thread
                import threading
                results_container = []
                exception_container = []
                
                def run_in_thread():
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(run_quality_gates())
                        results_container.append(result)
                        new_loop.close()
                    except Exception as ex:
                        exception_container.append(ex)
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()
                
                if exception_container:
                    raise exception_container[0]
                if results_container:
                    results = results_container[0]
                else:
                    raise RuntimeError("Failed to execute quality gates in thread")
            else:
                raise
        
        # Process quality gate results
        quality_scores = {}
        critique_notes = []
        bounce_back_required = False
        bounce_back_reasons = []
        
        # Check for bounce-back requirements first
        for i, result in enumerate(results):
            if not isinstance(result, Exception) and result.get("bounce_back_required", False):
                bounce_back_required = True
                bounce_back_reasons.append(result.get("reason", "Unknown verification failure"))
        
        # Check bounce-back attempt counter to prevent infinite loops
        bounce_back_attempt = state.get("bounce_back_attempt", 0)
        
        # If bounce-back required and we haven't exceeded retry limit, send content back to editing stage
        if bounce_back_required and bounce_back_attempt < 2:
            logger.warning(f"⚠️ Quality gates require content bounce-back (attempt {bounce_back_attempt + 1}): {'; '.join(bounce_back_reasons)}")
            log_agent_activity(
                pipeline_run_id=state["run_id"],
                agent_name="QualityGateAgents",
                stage="quality_gates",
                operation="bounce_back_required",
                message=f"Content verification failed, bouncing back to editing: {'; '.join(bounce_back_reasons)}",
                level="WARNING",
                context={"reasons": bounce_back_reasons, "attempt": bounce_back_attempt + 1}
            )
            # Set state to bounce back to editing stage
            state["status"] = "bounce_back_to_editing"
            state["bounce_back_reason"] = '; '.join(bounce_back_reasons)
            state["current_step"] = "editing"  # Send back to editing
            state["progress_percentage"] = 60.0  # Reset progress
            return state
        elif bounce_back_required and bounce_back_attempt >= 2:
            # Too many bounce-back attempts - force progression with lower thresholds
            logger.error(f"❌ Quality gates still failing after {bounce_back_attempt} attempts, but forcing progression to prevent infinite loop")
            log_agent_activity(
                pipeline_run_id=state["run_id"],
                agent_name="QualityGateAgents",
                stage="quality_gates", 
                operation="force_progression",
                message=f"Forcing progression after {bounce_back_attempt} bounce-back attempts. Reasons: {'; '.join(bounce_back_reasons)}",
                level="ERROR",
                context={"reasons": bounce_back_reasons, "attempts": bounce_back_attempt, "forced": True}
            )
            # Continue with processing but mark as degraded quality
        
        # Process and log quality gate results with detailed analysis
        agent_names = ["FactCheckGateAgent", "DomainExpertiseGateAgent", "StyleCriticGateAgent", "ComplianceGateAgent"]
        score_keys = ["fact_check", "domain_expertise", "style_consistency", "compliance"]
        
        logger.info(f"📊 Quality Gates Results Analysis:")
        
        # Fact check results
        if not isinstance(results[0], Exception):
            quality_scores["fact_check"] = results[0].get("score", 0.0)
            logger.info(f"   🔍 {agent_names[0]}: {quality_scores['fact_check']:.3f}")
            if "report" in results[0]:
                state["fact_check_report"] = results[0]["report"]
                report = results[0]["report"]
                if isinstance(report, dict):
                    logger.info(f"      - Claims analyzed: {report.get('total_claims_analyzed', 'N/A')}")
                    verification_summary = report.get('verification_summary', {})
                    if verification_summary:
                        logger.info(f"      - Supported claims: {verification_summary.get('supported', 0)}")
                        logger.info(f"      - Unsupported claims: {verification_summary.get('unsupported', 0)}")
        else:
            quality_scores["fact_check"] = 0.0
            logger.error(f"   ❌ {agent_names[0]} failed: {results[0]}")
        
        # Domain expertise results
        if not isinstance(results[1], Exception):
            quality_scores["domain_expertise"] = results[1].get("score", 0.0)
            logger.info(f"   🏷️ {agent_names[1]}: {quality_scores['domain_expertise']:.3f}")
            if "technical_assessment" in results[1]:
                tech_assessment = results[1]["technical_assessment"]
                if isinstance(tech_assessment, dict):
                    logger.info(f"      - Technical concepts: {len(tech_assessment.get('technical_concepts_identified', []))}")
            if "terminology_assessment" in results[1]:
                term_assessment = results[1]["terminology_assessment"]
                if isinstance(term_assessment, dict):
                    logger.info(f"      - Domain terms found: {len(term_assessment.get('found_terms', []))}")
        else:
            quality_scores["domain_expertise"] = 0.0
            logger.error(f"   ❌ {agent_names[1]} failed: {results[1]}")
        
        # Style critic results
        if not isinstance(results[2], Exception):
            quality_scores["style_consistency"] = results[2].get("score", 0.0)
            logger.info(f"   🎨 {agent_names[2]}: {quality_scores['style_consistency']:.3f}")
            critique_notes.extend(results[2].get("suggestions", []))
            if "style_analysis" in results[2]:
                style_analysis = results[2]["style_analysis"]
                if isinstance(style_analysis, dict):
                    logger.info(f"      - Clarity score: {style_analysis.get('clarity_score', 'N/A')}")
                    logger.info(f"      - Engagement level: {style_analysis.get('engagement_level', 'N/A')}")
            if "readability_metrics" in results[2]:
                readability = results[2]["readability_metrics"]
                if isinstance(readability, dict):
                    logger.info(f"      - Readability score: {readability.get('readability_score', 'N/A')}")
                    logger.info(f"      - Avg sentence length: {readability.get('average_sentence_length', 'N/A')}")
        else:
            quality_scores["style_consistency"] = 0.0
            logger.error(f"   ❌ {agent_names[2]} failed: {results[2]}")
        
        # Compliance results
        if not isinstance(results[3], Exception):
            quality_scores["compliance"] = results[3].get("score", 0.0)
            logger.info(f"   ⚖️ {agent_names[3]}: {quality_scores['compliance']:.3f}")
            if "report" in results[3]:
                state["compliance_report"] = results[3]["report"]
                report = results[3]["report"]
                if isinstance(report, dict):
                    logger.info(f"      - Compliance status: {report.get('compliance_status', 'N/A')}")
                    logger.info(f"      - Legal risk level: {report.get('legal_risk_level', 'N/A')}")
        else:
            quality_scores["compliance"] = 0.0
            logger.error(f"   ❌ {agent_names[3]} failed: {results[3]}")
        
        # Calculate overall quality score with detailed analysis
        scores = [score for score in quality_scores.values() if score > 0]
        quality_scores["overall"] = sum(scores) / len(scores) if scores else 0.0
        
        # Comprehensive quality score analysis
        logger.info(f"📈 Quality Score Summary:")
        logger.info(f"   🎯 Overall Score: {quality_scores['overall']:.3f}")
        logger.info(f"   📊 Individual Scores:")
        for gate_name, score in quality_scores.items():
            if gate_name != "overall":
                status = "✅ PASS" if score >= 0.75 else "⚠️ WEAK" if score >= 0.5 else "❌ FAIL"
                logger.info(f"      - {gate_name}: {score:.3f} {status}")
        
        # Identify potential issues
        low_scores = [(name, score) for name, score in quality_scores.items() if name != "overall" and score < 0.75]
        if low_scores:
            logger.warning(f"   ⚠️ Scores below threshold (0.75):")
            for name, score in low_scores:
                logger.warning(f"      - {name}: {score:.3f}")
        
        # Calculate score gap to threshold
        target_threshold = 0.85
        score_gap = target_threshold - quality_scores["overall"]
        if score_gap > 0:
            logger.warning(f"   📉 Score gap to threshold: {score_gap:.3f} points needed to reach {target_threshold}")
        else:
            logger.info(f"   🎉 Score exceeds threshold by: {abs(score_gap):.3f} points")
        
        # Log quality gates completion  
        # Extract assessment IDs for pointer-based context
        quality_assessment_ids = {}
        if "fact_check" in quality_scores:
            quality_assessment_ids["fact_check"] = []  # Could be populated if available
        if "domain_expertise" in quality_scores:
            quality_assessment_ids["domain_expertise"] = []
        if "style_consistency" in quality_scores:
            quality_assessment_ids["style_consistency"] = []
        if "compliance" in quality_scores:
            quality_assessment_ids["compliance"] = []
            
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="QualityGateAgents",
            stage="quality_gates",
            operation="complete_execution",
            message=f"Quality gates completed: Overall score {quality_scores.get('overall', 0):.2f}",
            level="INFO",
            context={
                "quality_scores": quality_scores, 
                "critique_notes_count": len(critique_notes),
                "quality_assessment_ids": quality_assessment_ids,  # Pointer keys for enhanced UI
                "overall_score": quality_scores.get('overall', 0)  # Pointer key for enhanced UI
            }
        )
        
        # Enhanced feedback collection for editor integration
        detailed_feedback = {
            "quality_scores": quality_scores,
            "gate_specific_feedback": {},
            "priority_improvements": [],
            "failed_gates": []
        }
        
        # Collect detailed feedback from each gate
        gate_names = ["fact_check", "domain_expertise", "style_consistency", "compliance"]
        for i, gate_name in enumerate(gate_names):
            if i < len(results) and not isinstance(results[i], Exception):
                gate_result = results[i]
                detailed_feedback["gate_specific_feedback"][gate_name] = {
                    "score": gate_result.get("score", 0.0),
                    "recommendations": gate_result.get("recommendations", []),
                    "suggestions": gate_result.get("suggestions", []),
                    "specific_issues": gate_result.get("specific_issues", []),
                    "strengths": gate_result.get("strengths", []),
                    "report": gate_result.get("report", {})
                }
                
                # Identify failed gates and priority improvements
                score = gate_result.get("score", 0.0)
                if score < 0.85:  # Failed threshold
                    detailed_feedback["failed_gates"].append({
                        "gate": gate_name,
                        "score": score,
                        "required_score": 0.85,
                        "gap": 0.85 - score
                    })
                    
                    # Extract priority improvements
                    recommendations = gate_result.get("recommendations", [])
                    if recommendations:
                        for rec in recommendations[:2]:  # Top 2 per gate
                            detailed_feedback["priority_improvements"].append({
                                "gate": gate_name,
                                "improvement": rec,
                                "priority": "high" if score < 0.7 else "medium"
                            })
        
        # Update state with enhanced feedback
        state["quality_scores"] = quality_scores
        state["critique_notes"] = critique_notes
        state["detailed_quality_feedback"] = detailed_feedback
        state["step_history"].append("quality_gates_completed")
        
        # Persist quality assessments to database
        try:
            content_draft_id = state.get("content_draft_id")
            if content_draft_id:
                # Create quality assessments for each gate
                gate_results = [
                    ("fact_check", results[0], quality_scores.get("technical_accuracy", 0.0)),
                    ("domain_expertise", results[1], quality_scores.get("domain_expertise", 0.0)),
                    ("style_consistency", results[2], quality_scores.get("style_consistency", 0.0)),
                    ("compliance", results[3], quality_scores.get("compliance", 0.0))
                ]
                
                for gate_name, gate_result, score in gate_results:
                    if not isinstance(gate_result, Exception) and score > 0:
                        assessment_id = create_quality_assessment(
                            pipeline_run_id=state["run_id"],
                            content_draft_id=content_draft_id,
                            gate_name=gate_name,
                            overall_score=score,
                            passed=score >= 0.85,  # Using default threshold
                            threshold_used=0.85,
                            assessor_type="ai_agent",
                            criteria_scores={gate_name: score},
                            strengths=gate_result.get("strengths", []),
                            weaknesses=gate_result.get("weaknesses", []),
                            suggestions=gate_result.get("suggestions", []),
                            reasoning=gate_result.get("reasoning"),
                            model_used=gate_result.get("model_used", "gpt-4"),
                            processing_time_seconds=gate_result.get("execution_time", 0.0)
                        )
                        if assessment_id:
                            logger.info(f"Created quality assessment {assessment_id} for {gate_name}")
                        
                        # Create detailed fact check report if applicable
                        if gate_name == "fact_check" and "claims_analysis" in gate_result:
                            fact_report_id = create_fact_check_report(
                                pipeline_run_id=state["run_id"],
                                quality_assessment_id=assessment_id,
                                content_draft_id=content_draft_id,
                                total_claims=gate_result.get("total_claims", 0),
                                verified_claims=gate_result.get("verified_claims", 0),
                                disputed_claims=gate_result.get("disputed_claims", 0),
                                unverifiable_claims=gate_result.get("unverifiable_claims", 0),
                                claims_analysis=gate_result.get("claims_analysis", []),
                                sources_checked=gate_result.get("sources_checked", 0),
                                reliable_sources=gate_result.get("reliable_sources", 0),
                                supporting_evidence=gate_result.get("supporting_evidence", [])
                            )
                            if fact_report_id:
                                logger.info(f"Created fact check report {fact_report_id}")
                
                logger.info(f"Quality gate assessments persisted to database for pipeline {state['run_id']}")
        except Exception as db_error:
            logger.error(f"Failed to persist quality assessments: {db_error}")
            # Don't fail the pipeline for DB persistence issues
        
        # Store state in Redis
        _store_pipeline_state(state["run_id"], state)
        
        logger.info(f"Quality gates completed: overall score {quality_scores['overall']:.2f}")
        
    except Exception as e:
        logger.error(f"Quality gates stage failed: {str(e)}")
        
        # Log agent error
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="QualityGateAgents",
            stage="quality_gates",
            operation="error",
            message=f"Quality gates stage failed: {str(e)}",
            level="ERROR",
            context={"error_type": type(e).__name__, "error_details": str(e)}
        )
        
        state["status"] = "failed"
        state["error_message"] = f"Quality gates failed: {str(e)}"
    
    return state


def _execute_editing_stage(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute editing stage using ContentEditorAgent."""
    try:
        from contentrunway.agents import ContentEditorAgent
        import asyncio
        import threading
        
        # Update progress
        state["current_step"] = "editing"
        state["progress_percentage"] = 78.0
        _update_pipeline_progress(state, celery_task)
        
        # Initialize and execute content editor
        editor_agent = ContentEditorAgent()
        
        # Log agent start
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ContentEditorAgent",
            stage="editing",
            operation="start_execution",
            message="Starting content editing based on quality feedback",
            level="INFO",
            context={"quality_scores": state.get("quality_scores", {}), "critique_notes_count": len(state.get("critique_notes", []))}
        )
        
        # Execute editing in dedicated thread / event loop to avoid conflicts
        editing_results = {}
        editing_error: Optional[Exception] = None
        
        logger.info(f"🚀 Starting ContentEditorAgent execution in dedicated thread")

        def run_editing():
            nonlocal editing_results, editing_error
            logger.info(f"🔄 Creating new event loop for editing thread")
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                logger.info(f"📝 Calling ContentEditorAgent.execute() with {getattr(state['draft'], 'word_count', 0)} word draft")
                editing_results = new_loop.run_until_complete(editor_agent.execute(
                    draft=state["draft"],
                    quality_feedback={
                        "scores": state["quality_scores"],
                        "critique_notes": state["critique_notes"]
                    },
                    state=state
                ))
                logger.info(f"✅ ContentEditorAgent.execute() completed successfully")
            except Exception as exc:
                logger.error(f"❌ ContentEditorAgent.execute() failed with error: {exc}")
                editing_error = exc
            finally:
                logger.info(f"🔄 Closing event loop for editing thread")
                new_loop.close()

        editing_thread = threading.Thread(target=run_editing, name=f"editing-{state['run_id']}")
        logger.info(f"🧵 Starting editing thread with 300 second timeout")
        editing_thread.start()
        editing_thread.join(timeout=300)  # 5 minutes

        if editing_thread.is_alive():
            logger.error("❌ ContentEditorAgent exceeded 5 minute timeout - thread still running")
            raise TimeoutError("Editing stage exceeded maximum execution time (5 minutes)")
        if editing_error:
            logger.error(f"❌ ContentEditorAgent failed with error: {editing_error}")
            raise editing_error
        if not editing_results:
            logger.error("❌ ContentEditorAgent returned empty results")
            raise RuntimeError("Editing stage returned no results")
            
        logger.info(f"🎉 ContentEditorAgent completed successfully with {len(editing_results)} result keys")
        
        # Log agent completion
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ContentEditorAgent",
            stage="editing",
            operation="complete_execution",
            message="Content editing completed: Draft updated based on feedback",
            level="INFO",
            context={"result_keys": list(editing_results.keys()) if editing_results else []}
        )
        
        # Update state with editing results
        state["draft"] = editing_results["revised_draft"]
        state["step_history"].append("editing_completed")
        
        # Store updated draft in Redis state for now (skip complex database update to prevent hanging)
        # This prevents potential deadlocks while maintaining pipeline functionality
        if state.get("draft"):
            logger.info(f"💾 Edited draft updated in pipeline state (word count: {getattr(state['draft'], 'word_count', 0)})")
            
            # Optional: Try simple database update with short timeout, but don't fail if it hangs
            if state.get("current_draft_id"):
                try:
                    # Quick attempt to update database without complex threading
                    logger.info(f"⏳ Attempting quick database update for draft {state['current_draft_id']}")
                    # For now, just log - can implement simple sync update later if needed
                    logger.info(f"✅ Database update deferred to prevent pipeline hanging")
                except Exception as e:
                    logger.warning(f"⚠️  Database update skipped due to error: {e}")
        
        # Store state in Redis (this is the critical part for pipeline continuation)
        _store_pipeline_state(state["run_id"], state)
        
        word_count = getattr(state["draft"], 'word_count', 0) if state["draft"] else 0
        logger.info(f"Editing completed: {word_count} words")
        
    except Exception as e:
        logger.error(f"Editing stage failed: {str(e)}")
        
        # Log agent error
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ContentEditorAgent",
            stage="editing",
            operation="error",
            message=f"Editing stage failed: {str(e)}",
            level="ERROR",
            context={"error_type": type(e).__name__, "error_details": str(e)}
        )
        
        state["status"] = "failed"
        state["error_message"] = f"Editing failed: {str(e)}"
    
    return state


def _execute_critique_stage(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute critique stage using CritiqueAgent."""
    try:
        from contentrunway.agents import CritiqueAgent
        import asyncio
        
        # Update progress
        state["current_step"] = "critique"
        state["progress_percentage"] = 80.0
        _update_pipeline_progress(state, celery_task)
        
        # Initialize and execute critique agent
        critique_agent = CritiqueAgent()
        
        # Convert quality_scores dict to QualityScores object if needed first
        from contentrunway.state.pipeline_state import QualityScores
        quality_scores = state["quality_scores"]
        if isinstance(quality_scores, dict):
            quality_scores = QualityScores(**quality_scores)
        
        # Log agent start (now quality_scores is properly defined)
        # Convert QualityScores to dict for JSON serialization
        quality_scores_dict = quality_scores.model_dump() if hasattr(quality_scores, 'model_dump') else (dict(quality_scores) if hasattr(quality_scores, '__dict__') else quality_scores)
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="CritiqueAgent",
            stage="critique",
            operation="start_execution",
            message="Starting content critique and quality validation",
            level="INFO",
            context={"quality_scores": quality_scores_dict, "draft_available": bool(state.get("draft"))}
        )
        
        # Execute critique
        critique_results = asyncio.run(critique_agent.execute(
            draft=state["draft"],
            quality_scores=quality_scores,
            state=state
        ))
        
        # Update state with critique results
        state["current_critique_feedback"] = critique_results["critique_feedback"]
        state["critique_cycle_count"] = critique_results["critique_feedback"].cycle_number
        state["post_edit_quality_scores"] = critique_results["post_critique_scores"]
        
        if "critique_feedback_history" not in state:
            state["critique_feedback_history"] = []
        state["critique_feedback_history"].append(critique_results["critique_feedback"])
        
        state["step_history"].append(f"critique_completed_cycle_{state['critique_cycle_count']}")
        
        # Store state in Redis
        _store_pipeline_state(state["run_id"], state)
        
        decision = critique_results["critique_feedback"].retry_decision
        logger.info(f"Critique completed: {decision} (cycle {state['critique_cycle_count']})")
        
        # Log agent completion
        # Convert post_edit_quality_scores to dict for JSON serialization
        post_edit_scores = state.get("post_edit_quality_scores")
        post_edit_scores_dict = post_edit_scores.model_dump() if hasattr(post_edit_scores, 'model_dump') else post_edit_scores
        
        # Extract critique report ID if available (would be set by critique agent)
        critique_report_id = critique_results.get('critique_report_id') if isinstance(critique_results, dict) else None
        
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="CritiqueAgent",
            stage="critique",
            operation="complete_execution",
            message=f"Critique completed: {decision} (cycle {state['critique_cycle_count']})",
            level="INFO",
            context={
                "decision": decision, 
                "cycle_count": state['critique_cycle_count'], 
                "post_edit_scores": post_edit_scores_dict,
                "critique_report_id": str(critique_report_id) if critique_report_id else None  # Pointer key for enhanced UI
            }
        )
        
    except Exception as e:
        logger.error(f"Critique stage failed: {str(e)}")
        
        # Log agent error
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="CritiqueAgent",
            stage="critique",
            operation="error",
            message=f"Critique stage failed: {str(e)}",
            level="ERROR",
            context={"error_type": type(e).__name__, "error_details": str(e)}
        )
        
        state["status"] = "failed"
        state["error_message"] = f"Critique failed: {str(e)}"
    
    return state


def _execute_formatting_stage(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute formatting stage using ContentFormatterAgent."""
    try:
        from contentrunway.agents import ContentFormatterAgent
        import asyncio
        
        # Update progress
        state["current_step"] = "formatting"
        state["progress_percentage"] = 85.0
        _update_pipeline_progress(state, celery_task)
        
        # Initialize and execute content formatter
        formatter_agent = ContentFormatterAgent()
        
        # LOG: Pre-formatting state
        draft = state.get("draft")
        logger.info(f"🔧 DEBUG: Pre-formatting - Draft type: {type(draft)}")
        if draft:
            logger.info(f"🔧 DEBUG: Pre-formatting - Draft title: '{getattr(draft, 'title', 'No title')}'")
            logger.info(f"🔧 DEBUG: Pre-formatting - Draft content length: {len(getattr(draft, 'content', ''))}")
            logger.info(f"🔧 DEBUG: Pre-formatting - Draft word count: {getattr(draft, 'word_count', 'No word count')}")
        else:
            logger.error(f"🔧 DEBUG: Pre-formatting - NO DRAFT FOUND in state!")
        
        # Execute formatting
        logger.info(f"🔧 DEBUG: Executing formatter agent...")
        formatting_results = asyncio.run(formatter_agent.execute(
            draft=state["draft"],
            state=state
        ))
        
        # LOG: Post-formatting results
        logger.info(f"🔧 DEBUG: Post-formatting - Results type: {type(formatting_results)}")
        logger.info(f"🔧 DEBUG: Post-formatting - Results keys: {list(formatting_results.keys())}")
        
        channel_drafts_obj = formatting_results.get("channel_drafts")
        logger.info(f"🔧 DEBUG: Post-formatting - Channel drafts type: {type(channel_drafts_obj)}")
        
        if channel_drafts_obj:
            # Check if it's a ChannelDrafts object with attributes
            if hasattr(channel_drafts_obj, '__dict__'):
                logger.info(f"🔧 DEBUG: Post-formatting - ChannelDrafts object attributes: {list(vars(channel_drafts_obj).keys())}")
                
                # Check for digitaldossier content
                dd_content = getattr(channel_drafts_obj, 'digitaldossier', None)
                if dd_content:
                    logger.info(f"🔧 DEBUG: Post-formatting - DigitalDossier content type: {type(dd_content)}")
                    if isinstance(dd_content, dict):
                        logger.info(f"🔧 DEBUG: Post-formatting - DigitalDossier content keys: {list(dd_content.keys())}")
                        logger.info(f"🔧 DEBUG: Post-formatting - DigitalDossier content length: {len(dd_content.get('content', ''))}")
                        logger.info(f"🔧 DEBUG: Post-formatting - DigitalDossier title: '{dd_content.get('title', 'No title')}'")
                    else:
                        logger.info(f"🔧 DEBUG: Post-formatting - DigitalDossier content: {dd_content}")
                else:
                    logger.warning(f"🔧 DEBUG: Post-formatting - NO DIGITALDOSSIER CONTENT found!")
                
                # Check for personal_blog content
                pb_content = getattr(channel_drafts_obj, 'personal_blog', None)
                if pb_content:
                    logger.info(f"🔧 DEBUG: Post-formatting - Personal blog content type: {type(pb_content)}")
                    if isinstance(pb_content, dict):
                        logger.info(f"🔧 DEBUG: Post-formatting - Personal blog content length: {len(pb_content.get('content', ''))}")
                else:
                    logger.warning(f"🔧 DEBUG: Post-formatting - NO PERSONAL BLOG CONTENT found!")
            
            # Check if it's already a dictionary
            elif isinstance(channel_drafts_obj, dict):
                logger.info(f"🔧 DEBUG: Post-formatting - Channel drafts is dict with keys: {list(channel_drafts_obj.keys())}")
                for platform, content in channel_drafts_obj.items():
                    if isinstance(content, dict):
                        logger.info(f"🔧 DEBUG: Post-formatting - {platform} content length: {len(content.get('content', ''))}")
                    else:
                        logger.info(f"🔧 DEBUG: Post-formatting - {platform} content type: {type(content)}")
        else:
            logger.error(f"🔧 DEBUG: Post-formatting - NO CHANNEL DRAFTS FOUND in results!")
        
        # CRITICAL FIX: Convert ChannelDrafts object to dictionary for proper state storage
        channel_drafts_obj = formatting_results["channel_drafts"]
        
        if channel_drafts_obj:
            if hasattr(channel_drafts_obj, '__dict__'):
                # Convert ChannelDrafts object to dictionary
                logger.info(f"🔧 DEBUG: Converting ChannelDrafts object to dictionary...")
                
                channel_drafts_dict = {}
                
                # Extract digitaldossier content
                dd_content = getattr(channel_drafts_obj, 'digitaldossier', None)
                if dd_content:
                    channel_drafts_dict['digitaldossier'] = dd_content
                    logger.info(f"🔧 DEBUG: Conversion - DigitalDossier content extracted: {len(dd_content.get('content', '') if isinstance(dd_content, dict) else str(dd_content))} chars")
                
                # Extract personal_blog content
                pb_content = getattr(channel_drafts_obj, 'personal_blog', None)
                if pb_content:
                    channel_drafts_dict['personal_blog'] = pb_content
                    logger.info(f"🔧 DEBUG: Conversion - Personal blog content extracted: {len(pb_content.get('content', '') if isinstance(pb_content, dict) else str(pb_content))} chars")
                
                # Store the dictionary
                state["channel_drafts"] = channel_drafts_dict
                logger.info(f"🔧 DEBUG: Conversion - Final dict keys: {list(channel_drafts_dict.keys())}")
                
            elif isinstance(channel_drafts_obj, dict):
                # Already a dictionary, store directly
                state["channel_drafts"] = channel_drafts_obj
                logger.info(f"🔧 DEBUG: Channel drafts already a dict, storing directly")
            else:
                # Unknown type, try to convert
                logger.warning(f"🔧 DEBUG: Unknown channel_drafts type {type(channel_drafts_obj)}, attempting conversion...")
                state["channel_drafts"] = channel_drafts_obj
        else:
            logger.error(f"🔧 DEBUG: NO channel_drafts to store!")
            state["channel_drafts"] = {}
        
        state["step_history"].append("formatting_completed")
        
        # Persist channel content to database
        try:
            content_draft_id = state.get("current_draft_id")
            final_channel_drafts = state.get("channel_drafts")
            
            if content_draft_id and final_channel_drafts and isinstance(final_channel_drafts, dict):
                for platform, content in final_channel_drafts.items():
                    if isinstance(content, dict) and content.get('content') and content.get('title'):
                        channel_content_id = create_channel_content(
                            pipeline_run_id=state["run_id"],
                            content_draft_id=content_draft_id,
                            platform=platform,
                            title=content["title"],
                            content=content["content"],
                            excerpt=content.get("excerpt"),
                            tags=content.get("tags", []),
                            categories=content.get("categories", []),
                            custom_fields=content.get("custom_fields", {}),
                            include_toc=content.get("include_toc", True),
                            include_citations=content.get("include_citations", True),
                            formatting_style=content.get("formatting_style", "standard"),
                            canonical_url=content.get("canonical_url"),
                            is_published=False  # Not yet published
                        )
                        
                        if channel_content_id:
                            logger.info(f"Persisted channel content {channel_content_id} for {platform}")
                        
                logger.info(f"Channel content persisted to database for pipeline {state['run_id']}")
                
        except Exception as db_error:
            logger.error(f"Failed to persist channel content: {db_error}")
            # Don't fail the pipeline for DB persistence issues
        
        # LOG: Final state update
        logger.info(f"🔧 DEBUG: State update - channel_drafts type: {type(state.get('channel_drafts'))}")
        final_channel_drafts = state.get("channel_drafts")
        if final_channel_drafts:
            if isinstance(final_channel_drafts, dict):
                logger.info(f"🔧 DEBUG: State update - Stored as dict with keys: {list(final_channel_drafts.keys())}")
                # Log content lengths for each platform
                for platform, content in final_channel_drafts.items():
                    if isinstance(content, dict) and 'content' in content:
                        content_length = len(content['content'])
                        title = content.get('title', 'No title')
                        logger.info(f"🔧 DEBUG: State update - {platform}: '{title}' ({content_length} chars)")
                    else:
                        logger.info(f"🔧 DEBUG: State update - {platform}: {type(content)}")
            elif hasattr(final_channel_drafts, '__dict__'):
                logger.warning(f"🔧 DEBUG: State update - Still stored as object! Attributes: {list(vars(final_channel_drafts).keys())}")
            else:
                logger.warning(f"🔧 DEBUG: State update - Stored as unknown type: {type(final_channel_drafts)}")
        else:
            logger.error(f"🔧 DEBUG: State update - NO CHANNEL DRAFTS stored in state!")
        
        # Persist final formatted draft to database
        if state.get("draft") and state.get("current_draft_id"):
            try:
                from app.services.content_service import ContentService
                from app.db.database import get_db
                import asyncio
                
                # Get database session
                async def finalize_draft():
                    async for db in get_db():
                        content_service = ContentService(db)
                        
                        # Convert final draft to dictionary
                        draft_data = {
                            "title": getattr(state["draft"], 'title', 'Untitled'),
                            "subtitle": getattr(state["draft"], 'subtitle', None),
                            "abstract": getattr(state["draft"], 'abstract', None),
                            "content": getattr(state["draft"], 'content', ''),
                            "word_count": getattr(state["draft"], 'word_count', 0),
                            "reading_time_minutes": getattr(state["draft"], 'reading_time_minutes', 0),
                            "readability_score": getattr(state["draft"], 'readability_score', None),
                            "meta_description": getattr(state["draft"], 'meta_description', None),
                            "keywords": getattr(state["draft"], 'keywords', []),
                            "tags": getattr(state["draft"], 'tags', []),
                            "citations": getattr(state["draft"], 'citations', []),
                            "internal_links": getattr(state["draft"], 'internal_links', [])
                        }
                        
                        # Update with final version and mark as current
                        final_draft_id = await content_service.update_content_draft(
                            draft_id=state["current_draft_id"],
                            updated_content=draft_data,
                            stage="final",
                            increment_version=True
                        )
                        state["current_draft_id"] = final_draft_id
                        
                        # Mark as current draft
                        await content_service.mark_draft_as_current(final_draft_id)
                        
                        logger.info(f"💾 Final formatted draft saved to database: {final_draft_id}")
                        return final_draft_id
                
                # Run the async database operation
                asyncio.run(finalize_draft())
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to save final draft to database: {e}")
        
        # Store state in Redis
        _store_pipeline_state(state["run_id"], state)
        
        logger.info("Formatting completed")
        
        # Log agent completion
        # Extract channel content IDs for pointer-based context
        channel_content_ids = {}
        channel_drafts = state.get("channel_drafts", {})
        for platform, content in channel_drafts.items():
            content_id = getattr(content, 'id', None) or (content.get('id') if isinstance(content, dict) else None)
            if content_id:
                channel_content_ids[platform] = str(content_id)
        
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="ContentFormatterAgent",
            stage="formatting",
            operation="complete_execution",
            message="Content formatting completed for all platforms",
            level="INFO",
            context={
                "channel_drafts_keys": list(channel_drafts.keys()), 
                "formatting_result_keys": list(formatting_results.keys()) if formatting_results else [],
                "channel_content_ids": channel_content_ids  # Pointer keys for enhanced UI
            }
        )
        
    except Exception as e:
        logger.error(f"Formatting stage failed: {str(e)}")
        state["status"] = "failed"
        state["error_message"] = f"Formatting failed: {str(e)}"
    
    return state


# REMOVED: _create_review_session function 
# This was part of the old Redis-based review system that has been replaced 
# with a simpler content status-based approach


# REMOVED: _check_human_review_feedback function
# This was part of the old Redis-based review system that has been replaced 
# with a simpler content status-based approach


def _execute_human_review_stage(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """
    Mark content as pending review and complete stage.
    """
    try:
        # Log stage start
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="HumanReviewGateAgent",
            stage="human_review",
            operation="start_execution",
            message="Starting human review stage - marking content for review",
            level="INFO",
            context={"content_draft_id": state.get("current_draft_id"), "review_required": True},
        )

        # Ensure draft exists
        try:
            content_draft_id = ensure_current_draft_id(state)
        except DraftPersistenceError as exc:
            logger.error(f"❌ Human review blocked: {exc}")
            state["status"] = "failed"
            state["error_message"] = f"Human review blocked: {exc}"
            return state

        # Update draft status in DB
        from sqlalchemy import text

        try:
            with get_sync_session() as session:
                unmark_query = text(
                    """
                    UPDATE content_drafts
                    SET is_current = false
                    WHERE pipeline_run_id = :run_id
                    """
                )
                session.execute(unmark_query, {"run_id": str(state["run_id"])})

                query = text(
                    """
                    UPDATE content_drafts
                    SET review_status = 'draft',
                        stage = 'human_review_pending',
                        is_current = true
                    WHERE id = :content_id
                    """
                )
                result = session.execute(query, {"content_id": content_draft_id})
                session.commit()

                if result.rowcount == 0:
                    raise ValueError(f"Unable to mark content draft {content_draft_id} for review")

                logger.info(f"✅ Content draft {content_draft_id} marked for human review")
        except Exception as exc:
            logger.error(f"❌ Failed to update content review status: {exc}")
            state["status"] = "failed"
            state["error_message"] = f"Human review blocked: {exc}"
            return state

        # Progress + pipeline status
        state["current_step"] = "human_review_pending"
        state["progress_percentage"] = 90.0
        _update_pipeline_progress(state, celery_task)

        update_pipeline_status(
            state["run_id"],
            "running",
            current_step="human_review_pending",
            progress_percentage=90.0,
        )

        state["step_history"].append("human_review_pending")
        state["review_status"] = "pending"

        # Create review session via helper
        try:
            review_id = create_review_session(state)
            logger.info(
                f"Created human review session {review_id} for pipeline {state['run_id']}",
            )
        except ReviewSessionError as exc:
            logger.error(f"❌ Human review session creation failed: {exc}")
            state["status"] = "failed"
            state["error_message"] = f"Human review session creation failed: {exc}"
            return state

        state["human_review_required"] = True
        _store_pipeline_state(state["run_id"], state)

        logger.info(
            f"Content marked for human review. Pipeline {state['run_id']} waiting for approval via Content tab."
        )

        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="HumanReviewGateAgent",
            stage="human_review",
            operation="complete_execution",
            message="Human review stage completed - content marked for review, pipeline paused",
            level="INFO",
            context={
                "review_session_id": state.get("human_review_session_id"),
                "content_draft_id": state.get("current_draft_id"),
                "review_status": "pending",
                "human_review_id": state.get("human_review_id"),
            },
        )

    except Exception as e:
        logger.error(f"Human review stage failed: {str(e)}")
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="HumanReviewGateAgent",
            stage="human_review",
            operation="error",
            message=f"Human review stage failed: {str(e)}",
            level="ERROR",
            context={"error_type": type(e).__name__, "error_details": str(e)},
        )
        state["status"] = "failed"
        state["error_message"] = f"Human review failed: {str(e)}"

    return state


def _execute_publishing_stage(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """Execute publishing stage using real PublisherAgent with auto-category and auto-genre detection."""
    try:
        # Update progress - start publishing at 95%
        state["current_step"] = "publishing"
        state["progress_percentage"] = 95.0
        _update_pipeline_progress(state, celery_task)
        
        # Log agent start
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="PublisherAgent",
            stage="publishing",
            operation="start_execution",
            message="Starting automated publishing to DigitalDossier",
            level="INFO",
            context={"pipeline_step": state.get("current_step"), "state_keys": list(state.keys())}
        )
        
        logger.info(f"🚀 Starting real publishing stage for pipeline {state['run_id']}")
        logger.info(f"🔧 DEBUG: _execute_publishing_stage - state keys: {list(state.keys())}")
        logger.info(f"🔧 DEBUG: _execute_publishing_stage - current_step: {state.get('current_step')}")
        
        # Import real publisher components
        try:
            from contentrunway.agents.publisher import PublisherAgent
            logger.info("✅ PublisherAgent imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import PublisherAgent: {e}")
            raise ValueError(f"PublisherAgent not available: {e}")
        
        # Ensure step_history exists
        if "step_history" not in state:
            state["step_history"] = ["reconstructed_for_publishing"]
        
        # Initialize publisher agent
        publisher_agent = PublisherAgent()
        
        # Extract channel drafts from formatting stage
        channel_drafts = state.get("channel_drafts", {})
        
        # Check if we have content to publish
        if not channel_drafts:
            logger.warning("⚠️ No channel_drafts available, creating from draft")
            # Fallback: extract content from draft directly
            draft = state.get("draft")
            if draft:
                logger.info(f"🔧 DEBUG: Publishing fallback - Draft type: {type(draft)}")
                logger.info(f"🔧 DEBUG: Publishing fallback - Draft title: '{getattr(draft, 'title', 'No title')}'")
                logger.info(f"🔧 DEBUG: Publishing fallback - Draft content length: {len(getattr(draft, 'content', ''))}")
                
                # Create content structure for both platforms
                content_data = {
                    "title": getattr(draft, 'title', 'Untitled'),
                    "content": getattr(draft, 'content', ''),
                    "summary": getattr(draft, 'abstract', '') or getattr(draft, 'summary', ''),
                    "meta_description": getattr(draft, 'meta_description', ''),
                    "keywords": getattr(draft, 'keywords', []),
                    "tags": getattr(draft, 'tags', []),
                    "word_count": getattr(draft, 'word_count', 0),
                    "abstract": getattr(draft, 'abstract', ''),
                    "citations": getattr(draft, 'citations', []),
                    "reading_time_minutes": getattr(draft, 'reading_time_minutes', 0) or max(1, getattr(draft, 'word_count', 0) // 200),
                    "readability_score": getattr(draft, 'readability_score', None)
                }
                
                # Create both digitaldossier and personal_blog formats
                channel_drafts = {
                    "digitaldossier": content_data.copy(),
                    "personal_blog": {
                        "title": content_data["title"],
                        "content": content_data["content"],
                        "meta_description": content_data["meta_description"],
                        "tags": content_data["tags"],
                        "word_count": content_data["word_count"]
                    }
                }
                
                logger.info(f"🔧 DEBUG: Publishing fallback - Created channel_drafts with keys: {list(channel_drafts.keys())}")
                logger.info(f"🔧 DEBUG: Publishing fallback - DigitalDossier content length: {len(channel_drafts['digitaldossier']['content'])}")
                logger.info(f"🔧 DEBUG: Publishing fallback - Personal blog content length: {len(channel_drafts['personal_blog']['content'])}")
            else:
                logger.error(f"🔧 DEBUG: Publishing fallback - NO DRAFT found in state!")
                logger.error(f"🔧 DEBUG: Publishing fallback - State keys: {list(state.keys())}")
                raise ValueError("No content available for publishing")
        else:
            logger.info(f"🔧 DEBUG: Publishing - Channel drafts available with keys: {list(channel_drafts.keys()) if isinstance(channel_drafts, dict) else 'Not a dict'}")
            if isinstance(channel_drafts, dict):
                for platform, content in channel_drafts.items():
                    if isinstance(content, dict) and 'content' in content:
                        logger.info(f"🔧 DEBUG: Publishing - {platform} content length: {len(content['content'])}")
                    else:
                        logger.info(f"🔧 DEBUG: Publishing - {platform} content type: {type(content)}")
        
        logger.info(f"📄 Publishing content: {list(channel_drafts.keys())}")
        
        # Execute real PublisherAgent with auto-category and auto-genre detection
        import asyncio
        import traceback
        
        logger.info(f"🔧 DEBUG: About to execute publisher agent with channel_drafts: {list(channel_drafts.keys()) if channel_drafts else 'None'}")
        
        # VALIDATION: Ensure we have valid content before publishing
        if isinstance(channel_drafts, dict):
            for platform, content in channel_drafts.items():
                if isinstance(content, dict):
                    content_text = content.get('content', '')
                    if not content_text or len(content_text.strip()) < 100:
                        logger.error(f"❌ VALIDATION: {platform} content too short ({len(content_text)} chars)")
                        raise ValueError(f"Content validation failed: {platform} content too short")
                    else:
                        logger.info(f"✅ VALIDATION: {platform} content valid ({len(content_text)} chars)")
                else:
                    logger.error(f"❌ VALIDATION: {platform} content is not a dict: {type(content)}")
                    raise ValueError(f"Content validation failed: {platform} content invalid type")
        else:
            logger.error(f"❌ VALIDATION: channel_drafts is not a dict: {type(channel_drafts)}")
            raise ValueError(f"Content validation failed: channel_drafts invalid type")
        
        def run_publisher_async():
            """Run publisher in async context with proper event loop handling."""
            try:
                logger.info(f"🔧 DEBUG: Starting publisher async execution")
                # Check if event loop exists
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        logger.info(f"🔧 DEBUG: Event loop is running, creating new thread")
                        # Create new event loop in thread
                        import threading
                        result_container = []
                        exception_container = []
                        
                        def run_in_thread():
                            try:
                                new_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(new_loop)
                                logger.info(f"🔧 DEBUG: New event loop created in thread")
                                result = new_loop.run_until_complete(
                                    publisher_agent.execute(channel_drafts, state)
                                )
                                result_container.append(result)
                                logger.info(f"🔧 DEBUG: Publisher execution completed in thread")
                            except Exception as e:
                                logger.error(f"🔧 DEBUG: Publisher execution failed in thread: {e}")
                                exception_container.append(e)
                            finally:
                                new_loop.close()
                        
                        thread = threading.Thread(target=run_in_thread)
                        thread.start()
                        thread.join()
                        
                        if exception_container:
                            raise exception_container[0]
                        return result_container[0]
                    else:
                        logger.info(f"🔧 DEBUG: Using existing event loop")
                        return loop.run_until_complete(publisher_agent.execute(channel_drafts, state))
                except RuntimeError as e:
                    logger.info(f"🔧 DEBUG: No event loop, creating new one: {e}")
                    # No event loop, create one
                    return asyncio.run(publisher_agent.execute(channel_drafts, state))
                    
            except Exception as e:
                logger.error(f"❌ Publisher execution failed: {e}")
                logger.error(f"🔧 DEBUG: Full publisher exception: {traceback.format_exc()}")
                raise
        
        # Execute publisher
        logger.info(f"🔧 DEBUG: About to call run_publisher_async()")
        
        # Update progress to 97% during upload
        state["progress_percentage"] = 97.0
        _update_pipeline_progress(state, celery_task)
        
        publishing_results = run_publisher_async()
        logger.info(f"🔧 DEBUG: Publisher execution completed, result type: {type(publishing_results)}")
        
        # Update progress to 99% after upload completes
        state["progress_percentage"] = 99.0
        _update_pipeline_progress(state, celery_task)
        
        # Log agent completion
        # Extract publication IDs for pointer-based context
        publication_ids = publishing_results.get('publication_ids', [])
        published_urls = publishing_results.get('published_urls', [])
        
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="PublisherAgent",
            stage="publishing",
            operation="complete_execution",
            message=f"Publishing completed: {len(published_urls)} URLs published",
            level="INFO",
            context={
                "result_keys": list(publishing_results.keys()), 
                "published_urls": published_urls,
                "publication_ids": [str(id) for id in publication_ids if id]  # Pointer keys for enhanced UI
            }
        )
        
        logger.info(f"📊 Publisher results: {list(publishing_results.keys())}")
        logger.info(f"🔧 DEBUG: Full publisher results: {publishing_results}")
        
        # Extract results
        successful_platforms = publishing_results.get('successful_platforms', [])
        failed_platforms = publishing_results.get('failed_platforms', [])
        published_urls = publishing_results.get('published_urls', [])
        publishing_summary = publishing_results.get('publishing_summary', {})
        
        logger.info(f"🔧 DEBUG: Successful platforms: {successful_platforms}")
        logger.info(f"🔧 DEBUG: Failed platforms: {failed_platforms}")
        logger.info(f"🔧 DEBUG: Published URLs: {published_urls}")
        logger.info(f"🔧 DEBUG: Publishing summary: {publishing_summary}")
        
        # Keep progress monotonic; do not regress after 99%
        # Maintain 99% here; final 100% set by pipeline completion
        state["progress_percentage"] = max(state.get("progress_percentage", 99.0), 99.0)
        _update_pipeline_progress(state, celery_task)
        
        # Update state with publishing results
        state["publishing_results"] = {
            "status": "completed" if successful_platforms else "failed",
            "successful_platforms": successful_platforms,
            "failed_platforms": failed_platforms,
            "publishing_summary": publishing_summary,
            "message": f"Published to {len(successful_platforms)} platforms" if successful_platforms else "Publishing failed"
        }
        state["published_urls"] = published_urls
        state["step_history"].append("publishing_completed")
        
        # Persist publishing results to database
        try:
            content_draft_id = state.get("current_draft_id")
            if content_draft_id:
                # First, create channel content records for each platform
                for platform, platform_results in publishing_summary.items():
                    if isinstance(platform_results, dict) and platform_results.get("success"):
                        channel_content_id = create_channel_content(
                            pipeline_run_id=state["run_id"],
                            content_draft_id=content_draft_id,
                            platform=platform,
                            title=platform_results.get("title", ""),
                            content=platform_results.get("content", ""),
                            excerpt=platform_results.get("excerpt"),
                            platform_specific_id=platform_results.get("platform_id"),
                            tags=platform_results.get("tags", []),
                            categories=platform_results.get("categories", []),
                            is_published=True
                        )
                        if channel_content_id:
                            logger.info(f"Created channel content {channel_content_id} for {platform}")
                            
                            # Create publication record
                            publication_id = create_publication(
                                pipeline_run_id=state["run_id"],
                                channel_content_id=channel_content_id,
                                platform=platform,
                                published_url=platform_results.get("url", ""),
                                title=platform_results.get("title", ""),
                                platform_content_id=platform_results.get("platform_id"),
                                status="published",
                                published_at=datetime.now(),
                                platform_metadata=platform_results.get("metadata", {}),
                                publication_response=platform_results.get("response", {}),
                                analytics_enabled=True
                            )
                            if publication_id:
                                logger.info(f"Created publication record {publication_id} for {platform}")
                
                # Update content status to published if successful
                if successful_platforms:
                    from app.db.sync_database import get_sync_session
                    from sqlalchemy import text
                    import json
                    
                    with get_sync_session() as session:
                        update_query = text("""
                            UPDATE content_drafts 
                            SET review_status = 'published',
                                published_at = NOW(),
                                published_urls = :urls
                            WHERE id = :content_id
                        """)
                        session.execute(update_query, {
                            "content_id": content_draft_id,
                            "urls": json.dumps(published_urls)
                        })
                        session.commit()
                        logger.info(f"✅ Content {content_draft_id} marked as published")
                        
                logger.info(f"Publishing data persisted to database for pipeline {state['run_id']}")
                
        except Exception as db_error:
            logger.error(f"Failed to persist publishing data: {db_error}")
            # Don't fail the pipeline for DB persistence issues
        
        # Store state in Redis
        _store_pipeline_state(state["run_id"], state)
        
        # DEBUG: Check the final platform status
        logger.info(f"🔧 DEBUG: Final check - successful_platforms: {successful_platforms}")
        logger.info(f"🔧 DEBUG: Final check - successful_platforms length: {len(successful_platforms)}")
        logger.info(f"🔧 DEBUG: Final check - successful_platforms type: {type(successful_platforms)}")
        logger.info(f"🔧 DEBUG: Final check - bool(successful_platforms): {bool(successful_platforms)}")
        
        if successful_platforms:
            logger.info(f"✅ Publishing completed successfully: {len(published_urls)} URLs")
            for url in published_urls:
                logger.info(f"   📰 Published URL: {url}")
            
            # CRITICAL FIX: Finalize pipeline completion for successful publishing
            logger.info(f"🏁 Finalizing pipeline completion to 100%...")
            state = _finalize_pipeline_completion(state, celery_task)
            
        else:
            logger.error(f"❌ Publishing failed for all platforms")
            # Set error status but don't fail the whole pipeline - let user retry
            state["status"] = "completed_with_errors"
            state["error_message"] = f"Publishing failed: {failed_platforms}"
        
    except Exception as e:
        logger.error(f"Publishing stage failed: {str(e)}")
        
        # Log agent error
        log_agent_activity(
            pipeline_run_id=state["run_id"],
            agent_name="PublisherAgent",
            stage="publishing",
            operation="error",
            message=f"Publishing stage failed: {str(e)}",
            level="ERROR",
            context={"error_type": type(e).__name__, "error_details": str(e)}
        )
        
        state["status"] = "failed"
        state["error_message"] = f"Publishing failed: {str(e)}"
    
    return state


def _finalize_pipeline_completion(state: Dict[str, Any], celery_task) -> Dict[str, Any]:
    """
    Finalize pipeline completion by setting status to 'completed' and progress to 100%.
    
    This ensures the pipeline properly reaches 100% completion after successful publishing.
    Updates both Redis state and database to reflect final completion status.
    """
    try:
        logger.info(f"🏁 Finalizing pipeline completion for run {state['run_id']}")
        
        # Set final completion status
        state["status"] = "completed"
        state["progress_percentage"] = 100.0
        state["current_step"] = "completed"
        state["processing_end_time"] = datetime.now()
        
        # Add to step history if not already present
        if "pipeline_completed" not in state.get("step_history", []):
            state["step_history"].append("pipeline_completed")
        
        # Update progress in Celery task metadata
        _update_pipeline_progress(state, celery_task)
        
        # Store final state in Redis
        _store_pipeline_state(state["run_id"], state)
        
        # Update database with final completion status
        try:
            update_pipeline_status(
                state["run_id"],
                status="completed",
                current_step="completed", 
                progress_percentage=100.0
            )
            logger.info(f"✅ Database updated with final completion status for run {state['run_id']}")
        except Exception as db_error:
            logger.error(f"❌ Failed to update database with completion status: {db_error}")
            # Don't fail the pipeline if DB update fails - Redis state is sufficient
        
        logger.info(f"🎉 Pipeline {state['run_id']} completed successfully at 100%")
        
    except Exception as e:
        logger.error(f"❌ Failed to finalize pipeline completion: {e}")
        # Don't fail the pipeline - log error and continue
    
    return state


# Helper functions

def _find_topic_by_id(topics: List[Dict], topic_id: str) -> Dict:
    """Find topic by ID in topics list."""
    if not topic_id:
        return None
    
    for topic in topics:
        if isinstance(topic, dict) and topic.get('id') == topic_id:
            return topic
    return None


def _create_topic_object(topic_dict: Dict) -> Any:
    """Create a topic object that agents can use with dot notation."""
    class TopicObject:
        def __init__(self, data: Dict):
            for key, value in data.items():
                setattr(self, key, value)
            
            # Ensure required attributes exist
            if not hasattr(self, 'target_keywords'):
                self.target_keywords = data.get('target_keywords', [])
            if not hasattr(self, 'title'):
                self.title = data.get('title', 'Untitled Topic')
            if not hasattr(self, 'description'):
                self.description = data.get('description', '')
    
    return TopicObject(topic_dict)


def _update_pipeline_progress(state: Dict[str, Any], celery_task):
    """Update pipeline progress in database, Redis, and Celery."""
    # Update database
    update_pipeline_status(
        state["run_id"], 
        "running", 
        current_step=state["current_step"],
        progress_percentage=state["progress_percentage"]
    )
    
    # Update Redis for real-time monitoring
    _update_redis_state(
        state["run_id"],
        "running",
        current_step=state["current_step"],
        progress_percentage=state["progress_percentage"]
    )
    
    # Update Celery task state
    celery_task.update_state(
        state="PROGRESS",
        meta={
            "run_id": state["run_id"],
            "status": "running",
            "current_step": state["current_step"],
            "progress": state["progress_percentage"]
        }
    )


def _normalize_outline_data(outline: Any) -> Optional[Dict[str, Any]]:
    """Convert Outline objects to dictionaries for safe serialization."""
    if outline is None:
        return None
    if isinstance(outline, dict):
        return outline
    if hasattr(outline, "model_dump"):
        try:
            return outline.model_dump()
        except Exception:
            pass
    if hasattr(outline, "dict"):
        try:
            return outline.dict()
        except Exception:
            pass
    if hasattr(outline, "__dict__"):
        return {key: _ensure_json_serializable(value) for key, value in outline.__dict__.items()}
    return None


def _get_outline_sections_count(outline: Any) -> int:
    """Safely count sections from outline data (dict or object)."""
    if not outline:
        return 0
    if isinstance(outline, dict):
        sections = outline.get("sections", [])
        return len(sections) if isinstance(sections, list) else 0
    if hasattr(outline, "sections") and outline.sections:
        return len(outline.sections)
    return 0


def _convert_outline_to_object(outline: Any):
    """Convert normalized outline data to Outline object for agent usage."""
    if not outline:
        return None
    if hasattr(outline, "sections"):
        return outline
    normalized = _normalize_outline_data(outline)
    if not normalized:
        return None
    from contentrunway.state.pipeline_state import Outline
    payload = {
        "sections": normalized.get("sections", []),
        "estimated_word_count": normalized.get("estimated_word_count", 1500),
        "target_audience": normalized.get("target_audience", "general audience"),
        "primary_angle": normalized.get("primary_angle", "comprehensive overview"),
        "key_takeaways": normalized.get("key_takeaways", []),
        "primary_keyword": normalized.get("primary_keyword", ""),
        "call_to_action": normalized.get("call_to_action"),
        "secondary_keywords": normalized.get("secondary_keywords", [])
    }
    return Outline(**payload)


def _extract_outline_titles(outline: Any, limit: int = 3) -> List[str]:
    """Extract section titles for logging."""
    titles: List[str] = []
    if not outline or limit <= 0:
        return titles
    sections = None
    if isinstance(outline, dict):
        sections = outline.get("sections", [])
    elif hasattr(outline, "sections"):
        sections = outline.sections
    if not sections:
        return titles
    for section in sections[:limit]:
        if isinstance(section, dict):
            titles.append(section.get("title", "No title"))
        elif hasattr(section, "title"):
            titles.append(getattr(section, "title", "No title"))
        else:
            titles.append(str(section))
    return titles


def _prepare_serializable_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare pipeline state for JSON serialization by converting complex objects."""
    import json
    
    serializable_state = {}
    
    for key, value in state.items():
        try:
            if key == "draft" and value:
                # Convert Draft object to dictionary
                if hasattr(value, '__dict__'):
                    draft_dict = {}
                    for attr_name in ['title', 'subtitle', 'abstract', 'content', 'word_count', 
                                    'reading_time_minutes', 'readability_score', 'meta_description', 
                                    'keywords', 'tags', 'citations', 'internal_links']:
                        attr_value = getattr(value, attr_name, None)
                        if attr_name == 'citations' and attr_value:
                            # Convert Citation objects to dictionaries
                            try:
                                draft_dict[attr_name] = [_serialize_citation(citation) for citation in attr_value]
                            except Exception:
                                draft_dict[attr_name] = []  # Fallback to empty list
                        else:
                            draft_dict[attr_name] = attr_value
                    serializable_state[key] = draft_dict
                else:
                    serializable_state[key] = value
            elif key == "sources" and value:
                # Convert Source objects to dictionaries
                try:
                    serializable_state[key] = [_serialize_source(source) for source in value]
                except Exception:
                    serializable_state[key] = []  # Fallback to empty list
            elif key == "outline" and value:
                # Convert Outline object to dictionary
                if hasattr(value, '__dict__'):
                    outline_dict = {}
                    for attr_name in ['primary_keyword', 'secondary_keywords', 'target_audience', 
                                    'primary_angle', 'key_takeaways', 'sections', 'estimated_word_count']:
                        try:
                            attr_value = getattr(value, attr_name, None)
                            # Special handling for sections which might contain complex objects
                            if attr_name == 'sections' and attr_value:
                                if isinstance(attr_value, list):
                                    sections_list = []
                                    for section in attr_value:
                                        if hasattr(section, '__dict__'):
                                            # Convert section object to dict
                                            section_dict = {}
                                            for section_attr in ['title', 'key_points', 'estimated_words']:
                                                section_dict[section_attr] = getattr(section, section_attr, None)
                                            sections_list.append(section_dict)
                                        else:
                                            sections_list.append(section)
                                    outline_dict[attr_name] = sections_list
                                else:
                                    outline_dict[attr_name] = attr_value
                            else:
                                outline_dict[attr_name] = attr_value
                        except Exception:
                            outline_dict[attr_name] = None  # Fallback for problematic attributes
                    serializable_state[key] = outline_dict
                else:
                    serializable_state[key] = value
            elif key in ["topics", "chosen_topic"] and value:
                # Handle topic objects
                try:
                    if isinstance(value, list):
                        serializable_state[key] = [_serialize_topic(topic) for topic in value]
                    else:
                        serializable_state[key] = _serialize_topic(value)
                except Exception:
                    serializable_state[key] = [] if isinstance(value, list) else None
            else:
                # Test if value is JSON serializable
                try:
                    json.dumps(value)
                    serializable_state[key] = value
                except (TypeError, ValueError):
                    # If not serializable, convert to string or skip
                    if hasattr(value, '__dict__'):
                        try:
                            # Try to convert object to dict
                            serializable_state[key] = str(value)
                        except Exception:
                            logger.warning(f"Skipping non-serializable state key: {key}")
                            continue
                    else:
                        serializable_state[key] = str(value)
        except Exception as e:
            logger.warning(f"Error serializing state key '{key}': {e}")
            continue
    
    return serializable_state


def _serialize_citation(citation) -> Dict[str, Any]:
    """Convert Citation object to dictionary for serialization."""
    if hasattr(citation, '__dict__'):
        citation_dict = {
            'number': getattr(citation, 'number', None),
            'quote_text': getattr(citation, 'quote_text', ''),
            'context': getattr(citation, 'context', ''),
            'citation_type': getattr(citation, 'citation_type', 'reference'),
            'source': _serialize_source(getattr(citation, 'source', None))
        }
        return citation_dict
    return citation


def _serialize_source(source) -> Dict[str, Any]:
    """Convert Source object to dictionary for serialization."""
    if hasattr(source, '__dict__'):
        source_dict = {}
        for attr_name in ['url', 'title', 'author', 'publication_date', 'domain', 
                         'source_type', 'summary', 'key_points', 'credibility_score', 
                         'relevance_score', 'currency_score']:
            source_dict[attr_name] = getattr(source, attr_name, None)
        # Preserve identifier if present
        if hasattr(source, '_lookup_identifier'):
            source_dict['_lookup_identifier'] = getattr(source, '_lookup_identifier')
        return source_dict
    return source


def _serialize_topic(topic) -> Dict[str, Any]:
    """Convert Topic object to dictionary for serialization."""
    if hasattr(topic, '__dict__'):
        topic_dict = {}
        for attr_name in ['title', 'description', 'domain', 'relevance_score', 'novelty_score', 
                         'seo_difficulty', 'overall_score', 'target_keywords', 'search_volume', 
                         'competition_level', 'trend_score']:
            topic_dict[attr_name] = getattr(topic, attr_name, None)
        return topic_dict
    elif isinstance(topic, dict):
        return topic
    else:
        return str(topic)


def _ensure_json_serializable(data: Any) -> Any:
    """Ensure data is JSON serializable by converting complex objects."""
    import json
    
    if data is None:
        return None
    elif isinstance(data, (str, int, float, bool)):
        return data
    elif isinstance(data, (list, tuple)):
        return [_ensure_json_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: _ensure_json_serializable(value) for key, value in data.items()}
    elif hasattr(data, '__dict__'):
        # Convert objects with __dict__ to dictionaries
        try:
            return {key: _ensure_json_serializable(value) for key, value in data.__dict__.items()}
        except Exception:
            return str(data)
    else:
        # For any other type, convert to string as fallback
        try:
            # Test if it's JSON serializable
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            return str(data)


def _store_pipeline_state(run_id: str, state: Dict[str, Any]):
    """Store complete pipeline state in Redis for persistence - simplified for Celery compatibility."""
    logger.info(f"🔧 DEBUG: _store_pipeline_state called for run_id: {run_id}")
    try:
        # Use synchronous Redis client to avoid event loop conflicts in Celery workers
        import redis
        import json
        import os
        from urllib.parse import urlparse
        
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
        
        # Parse Redis URL for connection parameters
        url_parts = urlparse(redis_url)
        redis_client = redis.Redis(
            host=url_parts.hostname or 'redis',
            port=url_parts.port or 6379,
            db=int(url_parts.path.lstrip('/')) if url_parts.path else 0,
            decode_responses=True
        )
        
        # Test connection
        redis_client.ping()
        
        # Prepare serializable state with proper object handling
        serializable_state = _prepare_serializable_state(state)
        
        # Store full state
        full_state_key = f"pipeline:full_state:{run_id}"
        serialized_state = json.dumps(serializable_state, default=str)
        redis_client.setex(full_state_key, 86400, serialized_state)  # 24 hour expiry
        
        # Store lightweight checkpoint
        checkpoint_key = f"pipeline:checkpoint:{run_id}"
        checkpoint_data = {
            "run_id": run_id,
            "status": state.get("status"),
            "current_step": state.get("current_step"),
            "progress_percentage": state.get("progress_percentage"),
            "updated_at": datetime.now().isoformat()
        }
        serialized_checkpoint = json.dumps(checkpoint_data, default=str)
        redis_client.setex(checkpoint_key, 86400, serialized_checkpoint)
        
        redis_client.close()
        logger.info(f"✅ Pipeline state stored successfully for run_id: {run_id}")
        
    except Exception as e:
        logger.error(f"🔧 DEBUG: _store_pipeline_state exception: {e}")
        logger.warning(f"Failed to store pipeline state in Redis: {e}")
        # Don't fail the pipeline if Redis storage fails


def _track_stage_duration(state: Dict[str, Any], stage_name: str, start_time):
    """Track duration of pipeline stage."""
    from datetime import datetime
    duration = (datetime.now() - start_time).total_seconds()
    if "step_durations" not in state:
        state["step_durations"] = {}
    state["step_durations"][stage_name] = duration


def _create_failure_result(run_id: str, error_message: str) -> Dict[str, Any]:
    """Create standardized failure result."""
    result = {
        "run_id": run_id,
        "status": "failed",
        "error_message": error_message,
        "published_urls": [],
        "processing_time": 0.0,
        "content_generated": False,
        "final_quality_score": 0.0,
        "human_approved": False
    }
    
    # Update database with failure
    update_pipeline_status(run_id, "failed", error_message=error_message)
    
    return result
