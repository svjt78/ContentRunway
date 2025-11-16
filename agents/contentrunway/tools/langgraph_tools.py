"""
LangGraph Tools Integration Module

This module provides LangGraph-compatible tool definitions that bridge
the ContentRunway tool ecosystem with LangGraph pipeline execution.
"""

from .content_tools import (
    select_writing_style,
    analyze_content_quality,
    optimize_seo,
    validate_content,
    CONTENT_WRITING_TOOLS,
    CONTENT_VALIDATION_TOOLS,
    CONTENT_SEO_TOOLS
)

# LangGraph tool collections for different pipeline stages
LANGGRAPH_WRITING_TOOLS = CONTENT_WRITING_TOOLS
LANGGRAPH_VALIDATION_TOOLS = CONTENT_VALIDATION_TOOLS  
LANGGRAPH_SEO_TOOLS = CONTENT_SEO_TOOLS

# Export individual tool functions for direct import
__all__ = [
    "select_writing_style",
    "analyze_content_quality", 
    "optimize_seo",
    "validate_content",
    "LANGGRAPH_WRITING_TOOLS",
    "LANGGRAPH_VALIDATION_TOOLS",
    "LANGGRAPH_SEO_TOOLS"
]