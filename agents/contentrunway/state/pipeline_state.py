from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class Source(BaseModel):
    """Research source information."""
    url: str
    title: str
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    domain: str
    source_type: str  # article, paper, blog, etc.
    summary: str
    key_points: List[str] = []
    credibility_score: float
    relevance_score: float
    currency_score: float


class TopicIdea(BaseModel):
    """Generated topic idea with scoring."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    domain: str
    relevance_score: float
    novelty_score: float
    seo_difficulty: float
    overall_score: float
    target_keywords: List[str] = []
    estimated_traffic: Optional[int] = None
    competition_level: Optional[str] = None
    sources: List[Source] = []


class Outline(BaseModel):
    """Content outline structure."""
    sections: List[Dict[str, Any]]  # Hierarchical section structure
    estimated_word_count: int
    target_audience: str
    primary_angle: str
    key_takeaways: List[str]
    call_to_action: Optional[str] = None
    primary_keyword: str
    secondary_keywords: List[str] = []


class Citation(BaseModel):
    """Citation information."""
    number: int
    source: Source
    quote_text: str
    context: str
    citation_type: str  # direct_quote, paraphrase, data_point


class Draft(BaseModel):
    """Content draft with metadata."""
    title: str
    subtitle: Optional[str] = None
    abstract: Optional[str] = None
    content: str
    citations: List[Citation] = []
    word_count: int
    reading_time_minutes: int
    meta_description: Optional[str] = None
    keywords: List[str] = []
    tags: List[str] = []


class QualityScores(BaseModel):
    """Quality assessment scores from different gates."""
    fact_check: Optional[float] = None
    domain_expertise: Optional[float] = None
    style_consistency: Optional[float] = None
    technical_depth: Optional[float] = None
    compliance: Optional[float] = None
    overall: Optional[float] = None
    
    def calculate_overall(self) -> float:
        """Calculate overall score with weighted averaging and outlier protection."""
        scores = {
            'fact_check': self.fact_check,
            'domain_expertise': self.domain_expertise, 
            'style_consistency': self.style_consistency,
            'technical_depth': self.technical_depth,
            'compliance': self.compliance
        }
        
        # Filter valid scores
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        
        if not valid_scores:
            return 0.0
        
        # Weighted scoring - emphasize fact check and domain expertise
        weights = {
            'fact_check': 0.25,
            'domain_expertise': 0.25,
            'style_consistency': 0.20,
            'technical_depth': 0.15,
            'compliance': 0.15
        }
        
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        for category, score in valid_scores.items():
            weight = weights.get(category, 0.2)  # Default weight if category not found
            weighted_sum += score * weight
            total_weight += weight
        
        base_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Outlier protection: if one score is significantly lower, reduce impact
        score_values = list(valid_scores.values())
        if len(score_values) >= 3:
            sorted_scores = sorted(score_values)
            lowest_score = sorted_scores[0]
            median_score = sorted_scores[len(sorted_scores)//2]
            
            # If lowest score is significantly below median, apply gentle adjustment
            if median_score - lowest_score > 0.3:
                # Reduce the impact of the extreme outlier
                adjustment = min(0.05, (median_score - lowest_score - 0.3) * 0.1)
                base_score = min(1.0, base_score + adjustment)
        
        return base_score


class CritiqueFeedback(BaseModel):
    """Structured critique feedback for agent improvement."""
    cycle_number: int
    overall_score: float
    issues_identified: List[Dict[str, Any]] = []
    issues_resolved: List[Dict[str, Any]] = []
    issues_remaining: List[Dict[str, Any]] = []
    improvement_suggestions: List[Dict[str, Any]] = []
    retry_decision: str  # pass, retry, fail
    retry_reasoning: Optional[str] = None
    next_actions: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)


class ChannelDrafts(BaseModel):
    """Platform-specific content formatting."""
    personal_blog: Optional[Dict[str, Any]] = None


class HumanReviewFeedback(BaseModel):
    """Human review feedback and edits."""
    decision: str  # approved, rejected, needs_revision
    overall_rating: Optional[int] = None  # 1-5
    feedback_notes: Optional[str] = None
    inline_edits: List[Dict[str, Any]] = []
    quality_concerns: List[str] = []
    time_spent_seconds: Optional[int] = None


class PublishingResults(BaseModel):
    """Results from publishing to different platforms."""
    personal_blog: Optional[Dict[str, Any]] = None


class ContentPipelineState(TypedDict):
    """
    Complete state for the ContentRunway pipeline using LangGraph.
    This represents all data that flows through the agent workflow.
    
    Note: Using Annotated types with LangGraph reducers for proper state management.
    """
    # Core Pipeline Metadata
    run_id: str
    tenant_id: str  # Phase 1: always "personal"
    status: str  # initialized, running, paused, completed, failed
    created_at: datetime
    
    # Pipeline Configuration
    domain_focus: List[str]  # IT/Insurance, AI, Agentic AI, etc.
    quality_thresholds: Dict[str, float]  # Threshold scores for each quality gate
    
    # Research Phase Results
    research_query: Optional[str]
    sources: List[Source]
    topics: List[TopicIdea]
    chosen_topic_id: Optional[str]
    
    # Content Development
    outline: Optional[Outline]
    draft: Optional[Draft]
    channel_drafts: Optional[ChannelDrafts]
    
    # Quality Control
    quality_scores: QualityScores
    critique_notes: List[str]
    fact_check_report: Optional[Dict[str, Any]]
    compliance_report: Optional[Dict[str, Any]]
    
    # Critique System
    critique_cycle_count: int
    critique_feedback_history: List[CritiqueFeedback]
    current_critique_feedback: Optional[CritiqueFeedback]
    pre_edit_quality_scores: Optional[QualityScores]
    post_edit_quality_scores: Optional[QualityScores]
    
    # Human Review Process
    human_review_required: bool
    human_review_feedback: Optional[HumanReviewFeedback]
    review_session_url: Optional[str]
    
    # Publishing Results
    publishing_results: Optional[PublishingResults]
    published_urls: List[str]
    
    # Error Handling and Retry
    current_step: str
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    
    # Progress Tracking
    progress_percentage: float
    step_history: List[str]  # Track progression through pipeline steps
    
    # AI Model Usage Tracking
    llm_usage: Dict[str, Any]  # Token usage, costs, model selections
    
    # Agent Performance Tracking (for progressive training)
    agent_performance_metrics: Dict[str, Any]  # Performance data for each agent
    learning_data_quality: float  # Overall quality of training data collected (0.0-1.0)
    
    # Performance Metrics
    processing_start_time: Optional[datetime]
    processing_end_time: Optional[datetime]
    step_durations: Dict[str, float]  # Time taken for each step
    
    # Intermediate Results (for debugging and analysis)
    intermediate_results: Dict[str, Any]  # Store results from each agent
    
    # Configuration Overrides
    config_overrides: Dict[str, Any]  # Runtime configuration changes