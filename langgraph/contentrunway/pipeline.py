"""
ContentRunway LangGraph Pipeline - Main orchestration workflow.
"""

from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import ContentPipelineState
from .agents import (
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
    PublishingAgent
)


class ContentPipeline:
    """Main ContentRunway pipeline using LangGraph for orchestration."""
    
    def __init__(self, checkpointer_path: str = "pipeline_checkpoints.db"):
        """Initialize the content pipeline."""
        self.checkpointer = SqliteSaver.from_conn_string(checkpointer_path)
        self.graph = self._build_graph()
        
        # Initialize agents
        self.research_coordinator = ResearchCoordinatorAgent()
        self.content_curator = ContentCuratorAgent()
        self.seo_strategist = SEOStrategistAgent()
        self.content_writer = ContentWriterAgent()
        self.fact_check_gate = FactCheckGateAgent()
        self.domain_expertise_gate = DomainExpertiseGateAgent()
        self.style_critic_gate = StyleCriticGateAgent()
        self.compliance_gate = ComplianceGateAgent()
        self.content_editor = ContentEditorAgent()
        self.critique_agent = CritiqueAgent()
        self.content_formatter = ContentFormatterAgent()
        self.human_review_gate = HumanReviewGateAgent()
        self.publishing_agent = PublishingAgent()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for the pipeline."""
        workflow = StateGraph(ContentPipelineState)
        
        # Add nodes for each pipeline step
        workflow.add_node("research", self._research_step)
        workflow.add_node("curation", self._curation_step)
        workflow.add_node("seo_strategy", self._seo_strategy_step)
        workflow.add_node("writing", self._writing_step)
        workflow.add_node("quality_gates", self._quality_gates_step)
        workflow.add_node("editing", self._editing_step)
        workflow.add_node("critique", self._critique_step)
        workflow.add_node("formatting", self._formatting_step)
        workflow.add_node("human_review", self._human_review_step)
        workflow.add_node("publishing", self._publishing_step)
        workflow.add_node("completion", self._completion_step)
        
        # Set entry point
        workflow.set_entry_point("research")
        
        # Define the pipeline flow with conditional routing
        workflow.add_edge("research", "curation")
        workflow.add_edge("curation", "seo_strategy")
        workflow.add_edge("seo_strategy", "writing")
        workflow.add_edge("writing", "quality_gates")
        
        # Quality gate routing - if quality fails, go back to editing
        workflow.add_conditional_edges(
            "quality_gates",
            self._quality_gate_router,
            {
                "pass": "editing",  # Always go to editing first
                "fail": "editing",
                "retry": "writing"
            }
        )
        
        # Editing flow - goes to critique for validation
        workflow.add_edge("editing", "critique")
        
        # Critique routing - decides whether to retry editing or proceed
        workflow.add_conditional_edges(
            "critique",
            self._critique_router,
            {
                "pass": "formatting",
                "retry": "editing",
                "fail": "human_review"
            }
        )
        
        # Formatting to human review
        workflow.add_edge("formatting", "human_review")
        
        # Human review routing
        workflow.add_conditional_edges(
            "human_review",
            self._human_review_router,
            {
                "approved": "publishing",
                "needs_revision": "editing",
                "rejected": END
            }
        )
        
        # Publishing flow
        workflow.add_edge("publishing", "completion")
        workflow.add_edge("completion", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _research_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Execute research phase using research coordinator and domain agents."""
        print(f"🔍 Starting research phase for domains: {state['domain_focus']}")
        
        state["current_step"] = "research"
        state["status"] = "running"
        
        try:
            # Use research coordinator to orchestrate domain-specific research
            research_results = await self.research_coordinator.execute(
                query=state.get("research_query", ""),
                domains=state["domain_focus"],
                state=state
            )
            
            # Update state with research results
            state["sources"] = research_results["sources"]
            state["topics"] = research_results["topics"]
            state["progress_percentage"] = 20.0
            
            # Track step completion
            state["step_history"].append("research_completed")
            
            print(f"✅ Research completed: {len(state['sources'])} sources, {len(state['topics'])} topics")
            
        except Exception as e:
            state["error_message"] = f"Research failed: {str(e)}"
            state["status"] = "failed"
            print(f"❌ Research failed: {e}")
        
        return state
    
    async def _curation_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Execute topic curation and selection."""
        print("🎯 Starting topic curation")
        
        state["current_step"] = "curation"
        
        try:
            # Use content curator to score and select best topic
            curation_results = await self.content_curator.execute(
                topics=state["topics"],
                sources=state["sources"],
                state=state
            )
            
            # Update state with selected topic
            state["chosen_topic_id"] = curation_results["chosen_topic_id"]
            state["progress_percentage"] = 30.0
            
            state["step_history"].append("curation_completed")
            
            print(f"✅ Topic selected: {curation_results['chosen_topic_id']}")
            
        except Exception as e:
            state["error_message"] = f"Curation failed: {str(e)}"
            state["status"] = "failed"
            
        return state
    
    async def _seo_strategy_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Execute SEO strategy development."""
        print("📈 Starting SEO strategy")
        
        state["current_step"] = "seo_strategy"
        
        try:
            # Get selected topic
            chosen_topic = next(
                (t for t in state["topics"] if t.id == state["chosen_topic_id"]),
                None
            )
            
            if not chosen_topic:
                raise ValueError("No topic selected for SEO strategy")
            
            # Use SEO strategist to develop content strategy
            seo_results = await self.seo_strategist.execute(
                topic=chosen_topic,
                sources=state["sources"],
                state=state
            )
            
            # Update state with outline
            state["outline"] = seo_results["outline"]
            state["progress_percentage"] = 40.0
            
            state["step_history"].append("seo_strategy_completed")
            
            print(f"✅ SEO strategy completed with {len(state['outline'].sections)} sections")
            
        except Exception as e:
            state["error_message"] = f"SEO strategy failed: {str(e)}"
            state["status"] = "failed"
            
        return state
    
    async def _writing_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Execute content writing."""
        print("✍️ Starting content writing")
        
        state["current_step"] = "writing"
        
        try:
            # Use content writer to create draft
            writing_results = await self.content_writer.execute(
                outline=state["outline"],
                sources=state["sources"],
                state=state
            )
            
            # Update state with draft
            state["draft"] = writing_results["draft"]
            state["progress_percentage"] = 60.0
            
            state["step_history"].append("writing_completed")
            
            print(f"✅ Draft completed: {state['draft'].word_count} words")
            
        except Exception as e:
            state["error_message"] = f"Writing failed: {str(e)}"
            state["status"] = "failed"
            
        return state
    
    async def _quality_gates_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Execute parallel quality gates."""
        print("🛡️ Starting quality assessment")
        
        state["current_step"] = "quality_gates"
        
        try:
            # Execute quality gates in parallel
            quality_results = await self._run_quality_gates_parallel(state)
            
            # Update state with quality scores
            state["quality_scores"] = quality_results["scores"]
            state["fact_check_report"] = quality_results.get("fact_check_report")
            state["compliance_report"] = quality_results.get("compliance_report")
            state["critique_notes"] = quality_results.get("critique_notes", [])
            
            # Calculate overall quality score
            state["quality_scores"].overall = state["quality_scores"].calculate_overall()
            
            state["progress_percentage"] = 75.0
            state["step_history"].append("quality_gates_completed")
            
            print(f"✅ Quality assessment completed: {state['quality_scores'].overall:.2f}")
            
        except Exception as e:
            state["error_message"] = f"Quality gates failed: {str(e)}"
            state["status"] = "failed"
            
        return state
    
    async def _run_quality_gates_parallel(self, state: ContentPipelineState) -> Dict[str, Any]:
        """Run all quality gates in parallel for efficiency."""
        import asyncio
        from .state.pipeline_state import QualityScores
        
        # Create tasks for parallel execution
        tasks = [
            self.fact_check_gate.execute(state["draft"], state["sources"]),
            self.domain_expertise_gate.execute(state["draft"], state["domain_focus"]),
            self.style_critic_gate.execute(state["draft"], state),
            self.compliance_gate.execute(state["draft"])
        ]
        
        # Wait for all quality gates to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        quality_scores = QualityScores()
        critique_notes = []
        
        # Fact check results
        if not isinstance(results[0], Exception):
            quality_scores.fact_check = results[0]["score"]
            if "report" in results[0]:
                fact_check_report = results[0]["report"]
        
        # Domain expertise results
        if not isinstance(results[1], Exception):
            quality_scores.domain_expertise = results[1]["score"]
        
        # Style critic results
        if not isinstance(results[2], Exception):
            quality_scores.style_consistency = results[2]["score"]
            critique_notes.extend(results[2].get("suggestions", []))
        
        # Compliance results
        if not isinstance(results[3], Exception):
            quality_scores.compliance = results[3]["score"]
            if "report" in results[3]:
                compliance_report = results[3]["report"]
        
        return {
            "scores": quality_scores,
            "critique_notes": critique_notes,
            "fact_check_report": fact_check_report if 'fact_check_report' in locals() else None,
            "compliance_report": compliance_report if 'compliance_report' in locals() else None
        }
    
    def _quality_gate_router(self, state: ContentPipelineState) -> str:
        """Route based on quality gate results."""
        overall_threshold = state["quality_thresholds"].get("overall", 0.85)
        current_score = state["quality_scores"].overall or 0.0
        
        if current_score >= overall_threshold:
            return "pass"
        elif state["retry_count"] < state["max_retries"]:
            state["retry_count"] += 1
            return "retry" if current_score < (overall_threshold - 0.1) else "fail"
        else:
            return "fail"
    
    async def _editing_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Execute content editing based on quality feedback."""
        print("✏️ Starting content editing")
        
        state["current_step"] = "editing"
        
        try:
            # Use content editor to apply improvements
            editing_results = await self.content_editor.execute(
                draft=state["draft"],
                quality_feedback={
                    "scores": state["quality_scores"],
                    "critique_notes": state["critique_notes"]
                },
                state=state
            )
            
            # Update state with revised draft
            state["draft"] = editing_results["revised_draft"]
            
            state["step_history"].append("editing_completed")
            
            print(f"✅ Editing completed: {state['draft'].word_count} words")
            
        except Exception as e:
            state["error_message"] = f"Editing failed: {str(e)}"
            state["status"] = "failed"
            
        return state
    
    async def _critique_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Execute comprehensive critique analysis after editing."""
        print("🔍 Starting content critique")
        
        state["current_step"] = "critique"
        
        try:
            # Store pre-critique quality scores for comparison
            state["pre_edit_quality_scores"] = state["quality_scores"]
            
            # Execute critique analysis
            critique_results = await self.critique_agent.execute(
                draft=state["draft"],
                quality_scores=state["quality_scores"],
                state=state
            )
            
            # Update state with critique results
            state["current_critique_feedback"] = critique_results["critique_feedback"]
            state["critique_cycle_count"] = critique_results["critique_feedback"].cycle_number
            state["post_edit_quality_scores"] = critique_results["post_critique_scores"]
            
            # Add to critique feedback history
            if "critique_feedback_history" not in state:
                state["critique_feedback_history"] = []
            state["critique_feedback_history"].append(critique_results["critique_feedback"])
            
            # Store agent learning data for progressive improvement
            if critique_results.get("learning_data"):
                if "agent_performance_metrics" not in state:
                    state["agent_performance_metrics"] = {}
                state["agent_performance_metrics"][f"cycle_{state['critique_cycle_count']}"] = critique_results["learning_data"]
                state["learning_data_quality"] = critique_results["learning_data"].get("data_quality_score", 1.0)
            
            # Update progress
            if critique_results["retry_decision"] == "pass":
                state["progress_percentage"] = 80.0
            else:
                # Slight progress for critique completion
                state["progress_percentage"] = min(79.0, state["progress_percentage"] + 2.0)
            
            state["step_history"].append(f"critique_completed_cycle_{state['critique_cycle_count']}")
            
            print(f"✅ Critique completed: {critique_results['retry_decision']} (cycle {state['critique_cycle_count']})")
            
        except Exception as e:
            state["error_message"] = f"Critique failed: {str(e)}"
            state["status"] = "failed"
            print(f"❌ Critique failed: {e}")
        
        return state
    
    def _critique_router(self, state: ContentPipelineState) -> str:
        """Route based on critique analysis results."""
        if not state.get("current_critique_feedback"):
            # Fallback if critique failed
            return "fail"
        
        critique_feedback = state["current_critique_feedback"]
        decision = critique_feedback.retry_decision
        cycle_count = state.get("critique_cycle_count", 0)
        
        # Enforce maximum retry limit
        max_critique_cycles = 2
        if cycle_count >= max_critique_cycles and decision == "retry":
            print(f"⚠️  Maximum critique cycles ({max_critique_cycles}) reached, forcing human review")
            return "fail"
        
        # Log routing decision
        print(f"🎯 Critique routing: {decision} (cycle {cycle_count})")
        
        if decision == "pass":
            return "pass"
        elif decision == "retry":
            return "retry"
        else:  # "fail"
            return "fail"
    
    async def _formatting_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Execute platform-specific formatting."""
        print("🎨 Starting content formatting")
        
        state["current_step"] = "formatting"
        
        try:
            # Use content formatter to create platform variants
            formatting_results = await self.content_formatter.execute(
                draft=state["draft"],
                state=state
            )
            
            # Update state with channel drafts
            state["channel_drafts"] = formatting_results["channel_drafts"]
            state["progress_percentage"] = 85.0
            
            state["step_history"].append("formatting_completed")
            
            print("✅ Content formatting completed")
            
        except Exception as e:
            state["error_message"] = f"Formatting failed: {str(e)}"
            state["status"] = "failed"
            
        return state
    
    async def _human_review_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Execute human review gate."""
        print("👤 Starting human review")
        
        state["current_step"] = "human_review"
        state["human_review_required"] = True
        
        try:
            # Use human review gate to set up review session
            review_results = await self.human_review_gate.execute(
                draft=state["draft"],
                quality_scores=state["quality_scores"],
                state=state
            )
            
            # Update state with review session details
            state["review_session_url"] = review_results["review_url"]
            state["progress_percentage"] = 90.0
            
            state["step_history"].append("human_review_started")
            
            print(f"✅ Human review session created: {state['review_session_url']}")
            
            # Note: This step will wait for human input before proceeding
            # The human review feedback will be updated externally
            
        except Exception as e:
            state["error_message"] = f"Human review setup failed: {str(e)}"
            state["status"] = "failed"
            
        return state
    
    def _human_review_router(self, state: ContentPipelineState) -> str:
        """Route based on human review decision."""
        if not state.get("human_review_feedback"):
            # Still waiting for human review
            return "needs_revision"  # Default action
        
        decision = state["human_review_feedback"].decision
        
        if decision == "approved":
            return "approved"
        elif decision == "needs_revision":
            return "needs_revision" 
        else:  # rejected
            return "rejected"
    
    async def _publishing_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Execute publishing to multiple platforms."""
        print("🚀 Starting publishing")
        
        state["current_step"] = "publishing"
        
        try:
            # Use publishing agent to distribute content
            publishing_results = await self.publishing_agent.execute(
                channel_drafts=state["channel_drafts"],
                state=state
            )
            
            # Update state with publishing results
            state["publishing_results"] = publishing_results["results"]
            state["published_urls"] = publishing_results["urls"]
            state["progress_percentage"] = 95.0
            
            state["step_history"].append("publishing_completed")
            
            print(f"✅ Publishing completed: {len(state['published_urls'])} URLs")
            
        except Exception as e:
            state["error_message"] = f"Publishing failed: {str(e)}"
            state["status"] = "failed"
            
        return state
    
    async def _completion_step(self, state: ContentPipelineState) -> ContentPipelineState:
        """Complete the pipeline."""
        print("🎉 Pipeline completion")
        
        state["current_step"] = "completed"
        state["status"] = "completed"
        state["progress_percentage"] = 100.0
        state["step_history"].append("pipeline_completed")
        
        # Record completion time
        from datetime import datetime
        state["processing_end_time"] = datetime.now()
        
        print("✅ ContentRunway pipeline completed successfully!")
        
        return state
    
    async def execute_pipeline(self, initial_state: ContentPipelineState) -> ContentPipelineState:
        """Execute the complete content pipeline."""
        config = {"configurable": {"thread_id": initial_state["run_id"]}}
        
        # Execute the graph
        final_state = None
        async for state in self.graph.astream(initial_state, config):
            final_state = state
            
            # Emit progress updates (would integrate with Socket.IO)
            await self._emit_progress_update(state)
        
        return final_state
    
    async def _emit_progress_update(self, state: ContentPipelineState):
        """Emit progress updates for real-time monitoring."""
        # Store progress in Redis for real-time monitoring
        from app.services.redis_service import redis_service
        
        progress_data = {
            "run_id": state["run_id"],
            "status": state["status"], 
            "current_step": state["current_step"],
            "progress_percentage": state["progress_percentage"],
            "error_message": state.get("error_message"),
            "updated_at": datetime.now().isoformat()
        }
        
        # Store in Redis for real-time access
        await redis_service.store_pipeline_state(state["run_id"], state)
        
        # TODO: Emit to Socket.IO clients
        print(f"Progress: {progress_data}")