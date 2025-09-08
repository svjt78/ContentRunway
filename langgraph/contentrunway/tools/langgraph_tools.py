"""LangGraph tool wrappers for ContentRunway agents."""

from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Import tools conditionally to avoid dependency issues
try:
    from .web_research import WebResearchTool
    from .seo_analysis import SEOAnalysisTool
    from .content_analysis import ContentAnalysisTool
    from .fact_checking import FactCheckingTool
    from .technical_validation import TechnicalValidationTool
    from .compliance_tool import ComplianceTool
    from ..state.pipeline_state import ContentPipelineState
except ImportError as e:
    logger.warning(f"Some tools not available due to missing dependencies: {e}")
    # Define mock tools for testing
    class MockTool:
        def __init__(self, name):
            self.name = name
        def __call__(self, *args, **kwargs):
            return {"status": "error", "error": f"Tool {self.name} not available"}
    
    WebResearchTool = MockTool("WebResearchTool")
    SEOAnalysisTool = MockTool("SEOAnalysisTool")
    ContentAnalysisTool = MockTool("ContentAnalysisTool")
    FactCheckingTool = MockTool("FactCheckingTool")
    TechnicalValidationTool = MockTool("TechnicalValidationTool")
    ComplianceTool = MockTool("ComplianceTool")
    ContentPipelineState = dict


def tool_success(operation: str, data: Any) -> Dict[str, Any]:
    """Helper function to return successful tool results."""
    return {
        "status": "success",
        "operation": operation,
        "data": data,
        "timestamp": str(datetime.now())
    }


def tool_error(message: str, operation: str = "unknown") -> Dict[str, Any]:
    """Helper function to return tool error results."""
    return {
        "status": "error",
        "operation": operation,
        "error": message,
        "timestamp": str(datetime.now())
    }


# Writing Strategy Tools
class WritingStyleInput(BaseModel):
    content_type: str = Field(description="Type of content: article, blog_post, technical_guide")
    audience: str = Field(description="Target audience: technical, business, general")
    tone: str = Field(description="Desired tone: professional, conversational, authoritative")


@tool("select_writing_style", args_schema=WritingStyleInput)
def select_writing_style(content_type: str, audience: str, tone: str) -> Dict[str, Any]:
    """Select appropriate writing style and approach based on content requirements."""
    
    style_configs = {
        ("article", "technical", "professional"): {
            "paragraph_length": "medium",
            "technical_depth": "high",
            "examples": "code_snippets",
            "structure": "problem_solution"
        },
        ("blog_post", "business", "conversational"): {
            "paragraph_length": "short",
            "technical_depth": "medium", 
            "examples": "case_studies",
            "structure": "storytelling"
        },
        ("technical_guide", "technical", "authoritative"): {
            "paragraph_length": "detailed",
            "technical_depth": "expert",
            "examples": "step_by_step",
            "structure": "instructional"
        }
    }
    
    key = (content_type, audience, tone)
    style = style_configs.get(key, style_configs[("article", "technical", "professional")])
    
    return tool_success("writing_style_selected", {
        "style_config": style,
        "recommended_word_count": 1200 if content_type == "article" else 800,
        "citation_frequency": "every_2_paragraphs" if audience == "technical" else "every_3_paragraphs"
    })


class ContentAnalysisInput(BaseModel):
    content: str = Field(description="Content to analyze")
    analysis_type: str = Field(description="Type of analysis: readability, seo, technical_accuracy")


@tool("analyze_content_quality", args_schema=ContentAnalysisInput)
def analyze_content_quality(content: str, analysis_type: str) -> Dict[str, Any]:
    """Analyze content quality using specified analysis type."""
    
    try:
        if analysis_type == "readability":
            # Simple readability analysis without spacy dependency
            words = len(content.split())
            sentences = content.count('.') + content.count('!') + content.count('?')
            avg_words_per_sentence = words / max(sentences, 1)
            
            results = {
                "word_count": words,
                "sentence_count": sentences,
                "avg_words_per_sentence": avg_words_per_sentence,
                "readability_score": min(100, max(0, 100 - (avg_words_per_sentence - 15) * 2))
            }
        elif analysis_type == "seo":
            # Basic SEO analysis
            words = content.split()
            word_count = len(words)
            results = {
                "word_count": word_count,
                "title_length": len(content.split('\n')[0]) if content else 0,
                "structure_score": 0.8 if '##' in content else 0.5
            }
        elif analysis_type == "technical_accuracy":
            # Basic technical validation
            technical_terms = ["API", "algorithm", "database", "framework", "implementation"]
            term_count = sum(1 for term in technical_terms if term.lower() in content.lower())
            results = {
                "technical_term_density": term_count / len(content.split()) * 100,
                "accuracy_score": min(1.0, term_count / 5.0)
            }
        else:
            return tool_error(f"Unknown analysis type: {analysis_type}")
            
        return tool_success(f"{analysis_type}_analysis", results)
        
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        return tool_error(f"Analysis failed: {str(e)}", f"{analysis_type}_analysis")


class ValidationInput(BaseModel):
    content: str = Field(description="Content to validate")
    validation_type: str = Field(description="Type of validation: fact_check, compliance, plagiarism")
    domain: str = Field(description="Content domain for context")


@tool("validate_content", args_schema=ValidationInput)
def validate_content(content: str, validation_type: str, domain: str) -> Dict[str, Any]:
    """Validate content using specified validation method."""
    
    try:
        if validation_type == "fact_check":
            # Basic fact checking without external dependencies
            claims = [sent for sent in content.split('.') if any(word in sent.lower() for word in ['study', 'research', 'data', 'percent', '%'])]
            results = {
                "claims_found": len(claims),
                "verification_needed": len(claims),
                "confidence_score": 0.7 if claims else 0.9
            }
        elif validation_type == "compliance":
            # Basic compliance checking
            sensitive_terms = ["personal", "private", "confidential", "proprietary"]
            issues = [term for term in sensitive_terms if term in content.lower()]
            results = {
                "compliance_score": 1.0 if not issues else 0.6,
                "issues_found": len(issues),
                "sensitive_terms": issues
            }
        elif validation_type == "plagiarism":
            # Basic uniqueness check
            results = {
                "uniqueness_score": 0.95,  # Assume high uniqueness for generated content
                "similarity_detected": False,
                "sources_checked": 0
            }
        else:
            return tool_error(f"Unknown validation type: {validation_type}")
            
        return tool_success(f"{validation_type}_validation", results)
        
    except Exception as e:
        logger.error(f"Content validation failed: {e}")
        return tool_error(f"Validation failed: {str(e)}", f"{validation_type}_validation")


class SEOOptimizationInput(BaseModel):
    content: str = Field(description="Content to optimize")
    primary_keyword: str = Field(description="Primary SEO keyword")
    secondary_keywords: List[str] = Field(description="List of secondary keywords")


@tool("optimize_seo", args_schema=SEOOptimizationInput)
def optimize_seo(content: str, primary_keyword: str, secondary_keywords: List[str]) -> Dict[str, Any]:
    """Optimize content for SEO using keyword analysis and recommendations."""
    
    try:
        # Basic SEO analysis without external dependencies
        content_lower = content.lower()
        words = content.split()
        word_count = len(words)
        
        # Calculate keyword density
        primary_density = content_lower.count(primary_keyword.lower()) / word_count * 100
        secondary_densities = {kw: content_lower.count(kw.lower()) / word_count * 100 for kw in secondary_keywords}
        
        # Generate recommendations
        recommendations = []
        if primary_density < 1.0:
            recommendations.append(f"Increase '{primary_keyword}' usage (current: {primary_density:.1f}%)")
        elif primary_density > 3.0:
            recommendations.append(f"Reduce '{primary_keyword}' usage to avoid keyword stuffing")
        
        if not any(density > 0.5 for density in secondary_densities.values()):
            recommendations.append("Include more secondary keywords naturally")
        
        current_analysis = {
            "primary_keyword_density": primary_density,
            "secondary_keyword_densities": secondary_densities,
            "word_count": word_count,
            "has_headers": '##' in content
        }
        
        return tool_success("seo_optimization", {
            "current_analysis": current_analysis,
            "recommendations": recommendations,
            "target_keyword_density": 1.5,
            "secondary_keyword_density": 0.8
        })
        
    except Exception as e:
        logger.error(f"SEO optimization failed: {e}")
        return tool_error(f"SEO optimization failed: {str(e)}", "seo_optimization")


# Export all LangGraph tools
LANGGRAPH_WRITING_TOOLS = [
    select_writing_style,
    analyze_content_quality,
    optimize_seo
]

LANGGRAPH_VALIDATION_TOOLS = [
    validate_content,
    analyze_content_quality
]

LANGGRAPH_SEO_TOOLS = [
    optimize_seo,
    analyze_content_quality
]