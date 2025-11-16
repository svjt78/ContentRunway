"""Quality Threshold Optimizer for Cost-Efficient Quality Gates."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from ..state.pipeline_state import QualityScores, ContentPipelineState

logger = logging.getLogger(__name__)


class QualityGateStrategy(Enum):
    """Quality gate execution strategies."""
    FULL_GATES = "full_gates"           # Run all 4 quality gates
    SMART_SKIP = "smart_skip"           # Skip some gates based on initial scores
    PROGRESSIVE = "progressive"         # Run gates progressively based on confidence
    EARLY_TERMINATION = "early_termination"  # Terminate early if quality is very high


@dataclass
class QualityThresholds:
    """Quality thresholds for different optimization strategies."""
    # Overall thresholds
    pass_threshold: float = 0.85
    excellent_threshold: float = 0.95
    
    # Individual gate thresholds
    fact_check_threshold: float = 0.90
    domain_expertise_threshold: float = 0.90
    style_consistency_threshold: float = 0.88
    compliance_threshold: float = 0.95
    
    # Early termination thresholds
    skip_remaining_threshold: float = 0.95
    progressive_confidence_threshold: float = 0.90


@dataclass
class QualityOptimizationResult:
    """Result of quality optimization analysis."""
    strategy: QualityGateStrategy
    gates_to_run: List[str]
    gates_to_skip: List[str]
    estimated_cost_reduction: float
    confidence_level: str
    reasoning: str


class QualityOptimizer:
    """
    Optimizes quality gate execution for cost efficiency while maintaining quality standards.
    """
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
        
        # Gate execution costs (relative)
        self.gate_costs = {
            "fact_check": 1.0,      # Baseline cost
            "domain_expertise": 1.0,
            "style_critic": 0.6,    # Cheaper with GPT-3.5-turbo
            "compliance": 0.6       # Cheaper with GPT-3.5-turbo
        }
        
        # Gate dependency mapping
        self.gate_dependencies = {
            "fact_check": [],                    # Independent
            "domain_expertise": [],             # Independent
            "style_critic": ["fact_check"],     # Can skip if fact_check is excellent
            "compliance": ["fact_check", "domain_expertise"]  # Can skip if both are excellent
        }
    
    def optimize_quality_gates(
        self, 
        content: str, 
        state: ContentPipelineState,
        previous_quality_scores: Optional[QualityScores] = None
    ) -> QualityOptimizationResult:
        """
        Determine optimal quality gate execution strategy.
        
        Args:
            content: Content to be analyzed
            state: Current pipeline state
            previous_quality_scores: Previous quality scores if available
            
        Returns:
            QualityOptimizationResult with optimization strategy
        """
        
        # Analyze content characteristics
        content_analysis = self._analyze_content_characteristics(content)
        
        # Check for previous quality scores
        if previous_quality_scores:
            return self._optimize_with_previous_scores(content_analysis, previous_quality_scores)
        
        # Determine strategy based on content analysis
        strategy = self._determine_initial_strategy(content_analysis, state)
        
        if strategy == QualityGateStrategy.FULL_GATES:
            return QualityOptimizationResult(
                strategy=strategy,
                gates_to_run=["fact_check", "domain_expertise", "style_critic", "compliance"],
                gates_to_skip=[],
                estimated_cost_reduction=0.0,
                confidence_level="standard",
                reasoning="Content requires full quality assessment"
            )
        
        elif strategy == QualityGateStrategy.SMART_SKIP:
            return self._optimize_smart_skip(content_analysis)
        
        elif strategy == QualityGateStrategy.PROGRESSIVE:
            return self._optimize_progressive(content_analysis)
        
        else:  # EARLY_TERMINATION
            return self._optimize_early_termination(content_analysis)
    
    def should_terminate_early(
        self, 
        partial_scores: Dict[str, float], 
        gates_completed: List[str]
    ) -> Tuple[bool, str]:
        """
        Determine if quality gate execution should terminate early.
        
        Args:
            partial_scores: Scores from completed gates
            gates_completed: List of gates that have been completed
            
        Returns:
            (should_terminate, reasoning)
        """
        
        if len(gates_completed) < 2:
            return False, "Need at least 2 gates completed for early termination assessment"
        
        # Check if first two gates (fact_check, domain_expertise) are excellent
        if ("fact_check" in partial_scores and "domain_expertise" in partial_scores):
            fact_check_score = partial_scores["fact_check"]
            domain_score = partial_scores["domain_expertise"]
            
            if (fact_check_score >= self.thresholds.skip_remaining_threshold and 
                domain_score >= self.thresholds.skip_remaining_threshold):
                
                estimated_overall = (fact_check_score + domain_score) / 2
                
                if estimated_overall >= self.thresholds.pass_threshold:
                    return True, f"Excellent scores in core gates (fact: {fact_check_score:.3f}, domain: {domain_score:.3f})"
        
        return False, "Scores do not meet early termination criteria"
    
    def _analyze_content_characteristics(self, content: str) -> Dict[str, Any]:
        """Analyze content to determine quality gate requirements."""
        
        analysis = {
            "length": len(content),
            "complexity": "medium",  # Default
            "domain_indicators": [],
            "fact_claim_density": "medium",
            "compliance_risk": "low"
        }
        
        # Length-based analysis
        if len(content) < 800:
            analysis["complexity"] = "low"
        elif len(content) > 2000:
            analysis["complexity"] = "high"
        
        # Domain indicators
        it_insurance_terms = ["insurance", "cybersecurity", "digital transformation", "compliance", "regulation"]
        ai_terms = ["artificial intelligence", "machine learning", "neural network", "algorithm", "model"]
        agentic_terms = ["agent", "langgraph", "multi-agent", "orchestration", "workflow"]
        
        content_lower = content.lower()
        for term in it_insurance_terms:
            if term in content_lower:
                analysis["domain_indicators"].append("it_insurance")
                break
                
        for term in ai_terms:
            if term in content_lower:
                analysis["domain_indicators"].append("ai")
                break
                
        for term in agentic_terms:
            if term in content_lower:
                analysis["domain_indicators"].append("agentic_ai")
                break
        
        # Fact claim density analysis
        fact_indicators = ["studies show", "research indicates", "according to", "statistics", "%", "survey", "report"]
        fact_count = sum(1 for indicator in fact_indicators if indicator in content_lower)
        
        if fact_count > 5:
            analysis["fact_claim_density"] = "high"
        elif fact_count < 2:
            analysis["fact_claim_density"] = "low"
        
        # Compliance risk analysis
        compliance_keywords = ["personal data", "privacy", "gdpr", "hipaa", "regulation", "legal", "lawsuit"]
        compliance_count = sum(1 for keyword in compliance_keywords if keyword in content_lower)
        
        if compliance_count > 2:
            analysis["compliance_risk"] = "high"
        elif compliance_count > 0:
            analysis["compliance_risk"] = "medium"
        
        return analysis
    
    def _determine_initial_strategy(
        self, 
        content_analysis: Dict[str, Any], 
        state: ContentPipelineState
    ) -> QualityGateStrategy:
        """Determine initial quality gate strategy based on content analysis."""
        
        # High-risk content requires full gates
        if (content_analysis["compliance_risk"] == "high" or 
            content_analysis["fact_claim_density"] == "high"):
            return QualityGateStrategy.FULL_GATES
        
        # Low complexity content can use progressive approach
        if (content_analysis["complexity"] == "low" and 
            content_analysis["compliance_risk"] == "low"):
            return QualityGateStrategy.PROGRESSIVE
        
        # Medium complexity can use smart skip
        if content_analysis["complexity"] == "medium":
            return QualityGateStrategy.SMART_SKIP
        
        # Default to full gates for safety
        return QualityGateStrategy.FULL_GATES
    
    def _optimize_with_previous_scores(
        self, 
        content_analysis: Dict[str, Any], 
        previous_scores: QualityScores
    ) -> QualityOptimizationResult:
        """Optimize based on previous quality scores."""
        
        # If previous scores were excellent, use early termination
        if previous_scores.overall and previous_scores.overall >= self.thresholds.excellent_threshold:
            return QualityOptimizationResult(
                strategy=QualityGateStrategy.EARLY_TERMINATION,
                gates_to_run=["fact_check", "domain_expertise"],  # Run core gates only
                gates_to_skip=["style_critic", "compliance"],
                estimated_cost_reduction=40.0,
                confidence_level="high",
                reasoning=f"Previous overall score was excellent ({previous_scores.overall:.3f})"
            )
        
        # If previous scores were good, use smart skip
        elif previous_scores.overall and previous_scores.overall >= self.thresholds.pass_threshold:
            return QualityOptimizationResult(
                strategy=QualityGateStrategy.SMART_SKIP,
                gates_to_run=["fact_check", "domain_expertise", "style_critic"],
                gates_to_skip=["compliance"],
                estimated_cost_reduction=25.0,
                confidence_level="medium",
                reasoning=f"Previous overall score was good ({previous_scores.overall:.3f})"
            )
        
        # Otherwise, run full gates
        return QualityOptimizationResult(
            strategy=QualityGateStrategy.FULL_GATES,
            gates_to_run=["fact_check", "domain_expertise", "style_critic", "compliance"],
            gates_to_skip=[],
            estimated_cost_reduction=0.0,
            confidence_level="standard",
            reasoning="Previous scores indicate need for comprehensive assessment"
        )
    
    def _optimize_smart_skip(self, content_analysis: Dict[str, Any]) -> QualityOptimizationResult:
        """Optimize using smart skip strategy."""
        
        gates_to_run = ["fact_check", "domain_expertise"]  # Always run core gates
        gates_to_skip = []
        
        # Skip style critic if content is short and simple
        if (content_analysis["length"] < 1000 and 
            content_analysis["complexity"] == "low"):
            gates_to_skip.append("style_critic")
        else:
            gates_to_run.append("style_critic")
        
        # Skip compliance if low risk
        if content_analysis["compliance_risk"] == "low":
            gates_to_skip.append("compliance")
        else:
            gates_to_run.append("compliance")
        
        cost_reduction = sum(self.gate_costs[gate] for gate in gates_to_skip) / sum(self.gate_costs.values()) * 100
        
        return QualityOptimizationResult(
            strategy=QualityGateStrategy.SMART_SKIP,
            gates_to_run=gates_to_run,
            gates_to_skip=gates_to_skip,
            estimated_cost_reduction=cost_reduction,
            confidence_level="medium",
            reasoning=f"Smart skip based on content analysis: {len(gates_to_skip)} gates skipped"
        )
    
    def _optimize_progressive(self, content_analysis: Dict[str, Any]) -> QualityOptimizationResult:
        """Optimize using progressive strategy."""
        
        # Start with core gates
        gates_to_run = ["fact_check", "domain_expertise"]
        
        # Conditionally add others based on initial results
        conditional_gates = []
        if content_analysis["complexity"] != "low":
            conditional_gates.append("style_critic")
        if content_analysis["compliance_risk"] != "low":
            conditional_gates.append("compliance")
        
        return QualityOptimizationResult(
            strategy=QualityGateStrategy.PROGRESSIVE,
            gates_to_run=gates_to_run,
            gates_to_skip=conditional_gates,  # Will be determined based on initial results
            estimated_cost_reduction=30.0,  # Estimated
            confidence_level="medium",
            reasoning="Progressive execution based on initial gate results"
        )
    
    def _optimize_early_termination(self, content_analysis: Dict[str, Any]) -> QualityOptimizationResult:
        """Optimize using early termination strategy."""
        
        return QualityOptimizationResult(
            strategy=QualityGateStrategy.EARLY_TERMINATION,
            gates_to_run=["fact_check", "domain_expertise"],
            gates_to_skip=["style_critic", "compliance"],
            estimated_cost_reduction=40.0,
            confidence_level="high",
            reasoning="Content characteristics support early termination after core gates"
        )
    
    def estimate_cost_savings(
        self, 
        optimization_result: QualityOptimizationResult
    ) -> Dict[str, float]:
        """Estimate cost savings from quality optimization."""
        
        total_cost = sum(self.gate_costs.values())
        saved_cost = sum(self.gate_costs[gate] for gate in optimization_result.gates_to_skip)
        
        return {
            "total_cost": total_cost,
            "saved_cost": saved_cost,
            "cost_reduction_percentage": (saved_cost / total_cost) * 100,
            "remaining_cost": total_cost - saved_cost
        }


# Global optimizer instance
quality_optimizer = QualityOptimizer()


def optimize_quality_gate_execution(
    content: str,
    state: ContentPipelineState,
    previous_quality_scores: Optional[QualityScores] = None
) -> QualityOptimizationResult:
    """
    Convenience function to optimize quality gate execution.
    
    Args:
        content: Content to be analyzed
        state: Current pipeline state
        previous_quality_scores: Previous quality scores if available
        
    Returns:
        QualityOptimizationResult with optimization strategy
    """
    return quality_optimizer.optimize_quality_gates(content, state, previous_quality_scores)


def should_skip_remaining_gates(
    partial_scores: Dict[str, float],
    gates_completed: List[str]
) -> Tuple[bool, str]:
    """
    Convenience function to check if remaining gates should be skipped.
    
    Args:
        partial_scores: Scores from completed gates
        gates_completed: List of gates that have been completed
        
    Returns:
        (should_skip, reasoning)
    """
    return quality_optimizer.should_terminate_early(partial_scores, gates_completed)