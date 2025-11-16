"""
Quality Analytics API endpoints for quality gate performance insights.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from app.services.quality_analytics_service import quality_analytics_service, QualityMetrics
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

class QualityOverviewResponse(BaseModel):
    """Quality overview response model."""
    overall_metrics: Dict[str, float]
    gate_performance: Dict[str, Dict[str, float]]
    trends: List[Dict[str, Any]]
    insights: Dict[str, Any]
    revision_patterns: Dict[str, Any]
    analysis_period_days: int
    generated_at: datetime

class GatePerformanceResponse(BaseModel):
    """Gate performance history response model."""
    gate_type: str
    analysis_period_days: int
    daily_performance: List[Dict[str, Any]]
    summary: Dict[str, float]

class RevisionAnalyticsResponse(BaseModel):
    """Revision analytics response model."""
    total_runs_analyzed: int
    revision_success_rate: float
    average_attempts_to_pass: float
    runs_requiring_multiple_revisions: int
    runs_requiring_multiple_revisions_rate: float
    analysis_period_days: int

@router.get("/overview", response_model=QualityOverviewResponse)
async def get_quality_overview(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze (1-365)")
) -> QualityOverviewResponse:
    """
    Get comprehensive quality overview and analytics.
    
    Args:
        days: Number of days to analyze (default: 30, max: 365)
    
    Returns:
        Comprehensive quality metrics and insights
    """
    try:
        logger.info(f"📊 API request: Quality overview for {days} days")
        
        metrics = quality_analytics_service.get_quality_overview(days=days)
        
        # Convert QualityMetrics to response format
        trends_data = []
        for trend in metrics.trends:
            trends_data.append({
                "gate_type": trend.gate_type,
                "period": trend.period,
                "average_score": trend.average_score,
                "pass_rate": trend.pass_rate,
                "total_assessments": trend.total_assessments,
                "improvement_rate": trend.improvement_rate
            })
        
        insights_data = {
            "weakest_gates": metrics.insights.weakest_gates,
            "strongest_gates": metrics.insights.strongest_gates,
            "improvement_opportunities": metrics.insights.improvement_opportunities,
            "revision_success_rate": metrics.insights.revision_success_rate,
            "average_attempts_to_pass": metrics.insights.average_attempts_to_pass
        }
        
        return QualityOverviewResponse(
            overall_metrics=metrics.overall_metrics,
            gate_performance=metrics.gate_performance,
            trends=trends_data,
            insights=insights_data,
            revision_patterns=metrics.revision_patterns,
            analysis_period_days=days,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to get quality overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate quality overview: {str(e)}")

@router.get("/gate/{gate_type}/performance", response_model=GatePerformanceResponse)
async def get_gate_performance_history(
    gate_type: str,
    days: int = Query(90, ge=1, le=365, description="Number of days to analyze (1-365)")
) -> GatePerformanceResponse:
    """
    Get detailed performance history for a specific quality gate.
    
    Args:
        gate_type: Type of quality gate (fact_check, domain_expertise, style_consistency, compliance)
        days: Number of days to analyze (default: 90, max: 365)
    
    Returns:
        Detailed gate performance history and summary
    """
    valid_gate_types = ["fact_check", "domain_expertise", "style_consistency", "compliance"]
    if gate_type not in valid_gate_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid gate_type. Must be one of: {', '.join(valid_gate_types)}"
        )
    
    try:
        logger.info(f"📈 API request: {gate_type} performance history for {days} days")
        
        performance_data = quality_analytics_service.get_gate_performance_history(
            gate_type=gate_type, 
            days=days
        )
        
        # Calculate summary statistics
        summary = {}
        if performance_data:
            summary = {
                "total_days_with_data": len(performance_data),
                "average_daily_score": sum(d.get("average_score", 0) for d in performance_data) / len(performance_data),
                "average_daily_assessments": sum(d.get("total_assessments", 0) for d in performance_data) / len(performance_data),
                "overall_pass_rate": sum(d.get("passed_count", 0) for d in performance_data) / sum(d.get("total_assessments", 1) for d in performance_data) * 100,
                "best_day_score": max(d.get("average_score", 0) for d in performance_data),
                "worst_day_score": min(d.get("average_score", 0) for d in performance_data)
            }
        
        return GatePerformanceResponse(
            gate_type=gate_type,
            analysis_period_days=days,
            daily_performance=performance_data,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Failed to get gate performance history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get gate performance history: {str(e)}")

@router.get("/revisions/analytics", response_model=RevisionAnalyticsResponse)
async def get_revision_analytics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze (1-365)")
) -> RevisionAnalyticsResponse:
    """
    Get revision patterns and success rate analytics.
    
    Args:
        days: Number of days to analyze (default: 30, max: 365)
    
    Returns:
        Revision analytics and patterns
    """
    try:
        logger.info(f"🔄 API request: Revision analytics for {days} days")
        
        revision_data = quality_analytics_service.get_revision_analytics(days=days)
        
        return RevisionAnalyticsResponse(
            total_runs_analyzed=revision_data.get("total_runs_analyzed", 0),
            revision_success_rate=revision_data.get("revision_success_rate", 0.0),
            average_attempts_to_pass=revision_data.get("average_attempts_to_pass", 1.0),
            runs_requiring_multiple_revisions=revision_data.get("runs_requiring_multiple_revisions", 0),
            runs_requiring_multiple_revisions_rate=revision_data.get("runs_requiring_multiple_revisions_rate", 0.0),
            analysis_period_days=days
        )
        
    except Exception as e:
        logger.error(f"Failed to get revision analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get revision analytics: {str(e)}")

@router.get("/gates/summary")
async def get_gates_summary(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze (1-90)")
) -> Dict[str, Any]:
    """
    Get a quick summary of all quality gates performance.
    
    Args:
        days: Number of days to analyze (default: 7, max: 90)
    
    Returns:
        Quick summary of gate performance
    """
    try:
        logger.info(f"⚡ API request: Gates summary for {days} days")
        
        metrics = quality_analytics_service.get_quality_overview(days=days)
        
        # Create simplified summary
        summary = {
            "analysis_period_days": days,
            "total_assessments": metrics.overall_metrics.get("total_assessments", 0),
            "overall_pass_rate": metrics.overall_metrics.get("pass_rate", 0.0),
            "gates": {}
        }
        
        for gate_type, performance in metrics.gate_performance.items():
            summary["gates"][gate_type] = {
                "average_score": performance.get("average_score", 0.0),
                "pass_rate": performance.get("pass_rate", 0.0),
                "total_assessments": performance.get("total_assessments", 0),
                "status": "excellent" if performance.get("pass_rate", 0) >= 90 else 
                          "good" if performance.get("pass_rate", 0) >= 75 else
                          "needs_improvement"
            }
        
        # Add top recommendations
        summary["top_recommendations"] = metrics.insights.improvement_opportunities[:3]
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get gates summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get gates summary: {str(e)}")

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for quality analytics service."""
    return {"status": "healthy", "service": "quality_analytics", "timestamp": datetime.now().isoformat()}