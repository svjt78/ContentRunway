"""Smart Model Selection Utility for Cost Optimization.

Updated: 2025-09-25
- Upgraded Tier 1 from GPT-4 to GPT-4o (88% cost savings)
- Upgraded Tier 3 from GPT-3.5-turbo to GPT-4o-mini (60-75% cost savings + better quality)
- Updated pricing data to reflect current OpenAI API rates (2025)
- Blended input/output token costs for simplified cost estimation
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import logging
from dataclasses import dataclass, replace
from copy import deepcopy

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tiers for cost optimization."""
    TIER_1 = "tier_1"  # GPT-4 for complex tasks
    TIER_2 = "tier_2"  # GPT-4o-mini for analytical tasks  
    TIER_3 = "tier_3"  # GPT-3.5-turbo for simple tasks


@dataclass
class ModelConfig:
    """Model configuration with cost optimization parameters."""
    model_name: str
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float
    tier: ModelTier
    use_cases: List[str]


class ModelSelector:
    """Smart model selection for cost-optimized LLM usage."""
    
    def __init__(self):
        # Cost-optimized model configurations with 2025 pricing
        # Pricing based on OpenAI API rates as of 2025-09-25
        # Input/Output token costs are blended for simplicity (weighted avg ~70% input, 30% output)
        self.model_configs = {
            ModelTier.TIER_1: ModelConfig(
                model_name="gpt-4o",  # Upgraded from gpt-4 for 88% cost savings
                max_tokens=3000,  # Reduced from 4000
                temperature=0.3,
                cost_per_1k_tokens=0.0045,  # GPT-4o blended rate ($2.50 input + $10.00 output weighted)
                tier=ModelTier.TIER_1,
                use_cases=["complex_reasoning", "creative_writing", "final_review"]
            ),
            ModelTier.TIER_2: ModelConfig(
                model_name="gpt-4o-mini",
                max_tokens=2000,  # Optimized for analytical tasks
                temperature=0.2,
                cost_per_1k_tokens=0.0003,  # GPT-4o-mini blended rate ($0.15 input + $0.60 output weighted)
                tier=ModelTier.TIER_2,
                use_cases=["analytical_tasks", "fact_verification", "technical_editing"]
            ),
            ModelTier.TIER_3: ModelConfig(
                model_name="gpt-4o-mini",  # Upgraded from gpt-3.5-turbo for better performance + cost savings
                max_tokens=1500,  # Sufficient for simple tasks
                temperature=0.1,
                cost_per_1k_tokens=0.0003,  # GPT-4o-mini blended rate (same as Tier 2)
                tier=ModelTier.TIER_3,
                use_cases=["rule_based_tasks", "classification", "formatting"]
            )
        }
        
        # Agent-to-tier mapping for cost optimization
        self.agent_tier_mapping = {
            # Tier 1: Complex tasks requiring GPT-4
            "ResearchCoordinatorAgent": ModelTier.TIER_1,
            "ContentWriterAgent": ModelTier.TIER_1,
            
            # Tier 2: Analytical tasks suitable for GPT-4o-mini
            "CritiqueAgent": ModelTier.TIER_2,  # Downgraded from Tier 1 for 93% cost savings
            "FactCheckGateAgent": ModelTier.TIER_2,
            "DomainExpertiseGateAgent": ModelTier.TIER_2,
            "ContentEditorAgent": ModelTier.TIER_2,
            "SEOStrategistAgent": ModelTier.TIER_2,
            "ContentCuratorAgent": ModelTier.TIER_2,
            
            # Tier 3: Simple tasks suitable for GPT-3.5-turbo
            "StyleCriticGateAgent": ModelTier.TIER_3,
            "ComplianceGateAgent": ModelTier.TIER_3,
            "ContentFormatterAgent": ModelTier.TIER_3,
            "CategoryClassifierAgent": ModelTier.TIER_3,
            "TitleGeneratorAgent": ModelTier.TIER_3,
            "GenreGeneratorAgent": ModelTier.TIER_3,
        }
        
        # Fallback configurations for high-demand periods
        self.fallback_models = {
            ModelTier.TIER_1: ["gpt-4o", "gpt-4", "gpt-4-turbo", "claude-3-sonnet"],
            ModelTier.TIER_2: ["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-haiku"],
            ModelTier.TIER_3: ["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-haiku"]
        }
    
    def get_model_config(
        self, 
        agent_name: str, 
        task_type: Optional[str] = None,
        quality_requirement: Optional[str] = None
    ) -> ModelConfig:
        """
        Get optimized model configuration for an agent.
        
        Args:
            agent_name: Name of the agent requesting model config
            task_type: Type of task (for override logic)
            quality_requirement: Quality requirement level
            
        Returns:
            ModelConfig with optimized settings
        """
        # COST OPTIMIZATION OVERRIDE: Force all agents to use GPT-4o-mini
        config = ModelConfig(
            model_name="gpt-4o-mini",
            max_tokens=2000,  # Reasonable default
            temperature=0.3,  # Balanced default
            cost_per_1k_tokens=0.0003,  # GPT-4o-mini cost
            tier=ModelTier.TIER_3,
            use_cases=["cost_optimization"]
        )
        
        # Apply agent-specific optimizations while maintaining GPT-4o-mini
        config = self._apply_agent_optimizations(config, agent_name)
        
        logger.info(f"COST OVERRIDE: Selected {config.model_name} for {agent_name} (FORCED GPT-4o-mini for cost optimization)")
        
        return config
    
    def _apply_overrides(
        self, 
        base_tier: ModelTier, 
        task_type: Optional[str], 
        quality_requirement: Optional[str]
    ) -> ModelTier:
        """Apply override logic for special cases."""
        
        # Quality-based overrides
        if quality_requirement == "critical":
            return ModelTier.TIER_1
        elif quality_requirement == "low":
            return ModelTier.TIER_3
            
        # Task-based overrides
        if task_type in ["final_review", "creative_writing"]:
            return ModelTier.TIER_1
        elif task_type in ["classification", "formatting"]:
            return ModelTier.TIER_3
            
        return base_tier
    
    def _apply_agent_optimizations(self, config: ModelConfig, agent_name: str) -> ModelConfig:
        """Apply agent-specific optimizations."""
        
        # Classification agents need fewer tokens
        if "Classifier" in agent_name or "Generator" in agent_name:
            config.max_tokens = min(config.max_tokens, 500)
            
        # Quality gates can use moderate token limits
        elif "Gate" in agent_name:
            config.max_tokens = min(config.max_tokens, 1500)
            
        # Content generation agents need more tokens
        elif agent_name in ["ContentWriterAgent", "ContentEditorAgent"]:
            config.max_tokens = max(config.max_tokens, 2500)
            
        return config
    
    def get_estimated_cost(
        self, 
        agent_name: str, 
        input_tokens: int, 
        estimated_output_tokens: int
    ) -> float:
        """Estimate cost for a given agent operation."""
        config = self.get_model_config(agent_name)
        total_tokens = input_tokens + estimated_output_tokens
        return (total_tokens / 1000) * config.cost_per_1k_tokens
    
    def compare_costs(self, agent_name: str, input_tokens: int) -> Dict[str, float]:
        """Compare costs across different model tiers."""
        costs = {}
        estimated_output = min(1000, input_tokens // 2)  # Rough estimate
        
        for tier in ModelTier:
            config = self.model_configs[tier]
            total_tokens = input_tokens + estimated_output
            costs[config.model_name] = (total_tokens / 1000) * config.cost_per_1k_tokens
            
        return costs
    
    def get_fallback_model(self, tier: ModelTier, failed_model: str) -> Optional[str]:
        """Get fallback model when primary model fails."""
        fallback_list = self.fallback_models.get(tier, [])
        
        # Return first fallback that's not the failed model
        for model in fallback_list:
            if model != failed_model:
                return model
                
        return None
    
    def log_cost_savings(self, agent_name: str, original_model: str = "gpt-4"):
        """Log potential cost savings from optimization."""
        optimized_config = self.get_model_config(agent_name)
        
        # Calculate savings compared to original model (default: legacy GPT-4 usage)
        original_costs = {
            "gpt-4": 0.045,  # Legacy GPT-4 blended rate ($30 input + $60 output weighted)
            "gpt-3.5-turbo": 0.0008,  # GPT-3.5-turbo blended rate ($0.50 input + $1.50 output weighted)
        }
        
        original_cost = original_costs.get(original_model, 0.045)
        optimized_cost = optimized_config.cost_per_1k_tokens
        
        if original_cost > optimized_cost:
            savings_percent = ((original_cost - optimized_cost) / original_cost) * 100
            logger.info(
                f"Cost optimization for {agent_name}: "
                f"{original_model} → {optimized_config.model_name} "
                f"({savings_percent:.1f}% savings)"
            )
        else:
            cost_increase_percent = ((optimized_cost - original_cost) / original_cost) * 100
            logger.info(
                f"Quality upgrade for {agent_name}: "
                f"{original_model} → {optimized_config.model_name} "
                f"({cost_increase_percent:.1f}% cost increase for better performance)"
            )


# Global instance for easy access
model_selector = ModelSelector()


def get_optimized_model_config(
    agent_name: str, 
    task_type: Optional[str] = None,
    quality_requirement: Optional[str] = None
) -> ModelConfig:
    """
    Convenience function to get optimized model configuration.
    
    Args:
        agent_name: Name of the agent
        task_type: Optional task type for overrides
        quality_requirement: Optional quality requirement
        
    Returns:
        Optimized ModelConfig
    """
    return model_selector.get_model_config(agent_name, task_type, quality_requirement)


def estimate_operation_cost(
    agent_name: str, 
    input_tokens: int, 
    estimated_output_tokens: int = None
) -> float:
    """
    Estimate cost for an agent operation.
    
    Args:
        agent_name: Name of the agent
        input_tokens: Number of input tokens
        estimated_output_tokens: Estimated output tokens (defaults to input/2)
        
    Returns:
        Estimated cost in dollars
    """
    if estimated_output_tokens is None:
        estimated_output_tokens = min(1000, input_tokens // 2)
        
    return model_selector.get_estimated_cost(agent_name, input_tokens, estimated_output_tokens)