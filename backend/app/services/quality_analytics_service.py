"""
Quality Analytics Service - Advanced analytics for quality gate performance and trends.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import text, func, and_, or_
from ..db.sync_database import get_sync_session

logger = logging.getLogger(__name__)

@dataclass
class QualityTrend:
    """Quality trend data over time."""
    gate_type: str
    period: str
    average_score: float
    pass_rate: float
    total_assessments: int
    improvement_rate: float  # Percentage change from previous period

@dataclass
class QualityInsights:
    """Quality insights and recommendations."""
    weakest_gates: List[str]
    strongest_gates: List[str]
    improvement_opportunities: List[str]
    revision_success_rate: float
    average_attempts_to_pass: float

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics."""
    overall_metrics: Dict[str, float]
    gate_performance: Dict[str, Dict[str, float]]
    trends: List[QualityTrend]
    insights: QualityInsights
    revision_patterns: Dict[str, Any]

class QualityAnalyticsService:
    """Service for quality gate analytics and insights."""
    
    def __init__(self):
        self.logger = logger
    
    def get_quality_overview(self, days: int = 30) -> QualityMetrics:
        """
        Get comprehensive quality overview for the last N days.
        
        Args:
            days: Number of days to analyze (default: 30)
        
        Returns:
            QualityMetrics with comprehensive analytics
        """
        self.logger.info(f"📊 Generating quality overview for last {days} days")
        
        try:
            with get_sync_session() as session:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Overall metrics
                overall_metrics = self._calculate_overall_metrics(session, start_date, end_date)
                
                # Gate-specific performance
                gate_performance = self._calculate_gate_performance(session, start_date, end_date)
                
                # Quality trends
                trends = self._calculate_quality_trends(session, start_date, end_date)
                
                # Quality insights
                insights = self._generate_quality_insights(session, start_date, end_date)
                
                # Revision patterns
                revision_patterns = self._analyze_revision_patterns(session, start_date, end_date)
                
                return QualityMetrics(
                    overall_metrics=overall_metrics,
                    gate_performance=gate_performance,
                    trends=trends,
                    insights=insights,
                    revision_patterns=revision_patterns
                )
                
        except Exception as e:
            self.logger.error(f"Failed to generate quality overview: {str(e)}")
            return self._get_default_metrics()
    
    def get_gate_performance_history(self, gate_type: str, days: int = 90) -> List[Dict[str, Any]]:
        """
        Get detailed performance history for a specific gate.
        
        Args:
            gate_type: Type of quality gate (fact_check, domain_expertise, etc.)
            days: Number of days to analyze
        
        Returns:
            List of daily performance data
        """
        self.logger.info(f"📈 Analyzing {gate_type} performance over {days} days")
        
        try:
            with get_sync_session() as session:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                query = text("""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as total_assessments,
                        AVG(score) as average_score,
                        COUNT(*) FILTER (WHERE passed = true) as passed_count,
                        AVG(score) FILTER (WHERE passed = true) as avg_passed_score,
                        AVG(score) FILTER (WHERE passed = false) as avg_failed_score,
                        AVG(execution_time_seconds) as avg_execution_time
                    FROM quality_assessments 
                    WHERE gate_type = :gate_type 
                        AND created_at >= :start_date 
                        AND created_at <= :end_date
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                """)
                
                result = session.execute(query, {
                    "gate_type": gate_type,
                    "start_date": start_date,
                    "end_date": end_date
                })
                
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            self.logger.error(f"Failed to get gate performance history: {str(e)}")
            return []
    
    def get_revision_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze revision patterns and success rates.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with revision analytics
        """
        self.logger.info(f"🔄 Analyzing revision patterns for last {days} days")
        
        try:
            with get_sync_session() as session:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Get pipeline runs with revision attempts
                query = text("""
                    SELECT 
                        pr.id as pipeline_run_id,
                        pr.status,
                        COUNT(qa.id) as total_assessments,
                        COUNT(qa.id) FILTER (WHERE qa.passed = false) as failed_assessments,
                        MIN(qa.created_at) as first_assessment,
                        MAX(qa.created_at) as last_assessment,
                        COUNT(DISTINCT qa.gate_type) as unique_gates_tested
                    FROM pipeline_runs pr
                    LEFT JOIN quality_assessments qa ON pr.id = qa.pipeline_run_id
                    WHERE pr.created_at >= :start_date 
                        AND pr.created_at <= :end_date
                    GROUP BY pr.id, pr.status
                    HAVING COUNT(qa.id) > 0
                """)
                
                result = session.execute(query, {
                    "start_date": start_date,
                    "end_date": end_date
                })
                
                runs_data = [dict(row._mapping) for row in result]
                
                # Analyze revision patterns
                revision_success_rate = 0.0
                avg_attempts = 0.0
                multiple_revision_runs = 0
                
                if runs_data:
                    successful_runs = [r for r in runs_data if r['status'] == 'completed']
                    revision_success_rate = len(successful_runs) / len(runs_data) * 100
                    
                    # Estimate revision attempts based on assessment counts
                    attempts = [max(1, r['total_assessments'] // 4) for r in runs_data]  # Assuming 4 gates per attempt
                    avg_attempts = sum(attempts) / len(attempts) if attempts else 1.0
                    
                    multiple_revision_runs = len([r for r in runs_data if r['total_assessments'] > 4])
                
                return {
                    "total_runs_analyzed": len(runs_data),
                    "revision_success_rate": revision_success_rate,
                    "average_attempts_to_pass": avg_attempts,
                    "runs_requiring_multiple_revisions": multiple_revision_runs,
                    "runs_requiring_multiple_revisions_rate": multiple_revision_runs / len(runs_data) * 100 if runs_data else 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to analyze revision patterns: {str(e)}")
            return {}
    
    def _calculate_overall_metrics(self, session: Session, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate overall quality metrics."""
        query = text("""
            SELECT 
                COUNT(*) as total_assessments,
                AVG(score) as average_score,
                COUNT(*) FILTER (WHERE passed = true) as passed_assessments,
                AVG(execution_time_seconds) as avg_execution_time,
                COUNT(DISTINCT pipeline_run_id) as unique_runs
            FROM quality_assessments 
            WHERE created_at >= :start_date AND created_at <= :end_date
        """)
        
        result = session.execute(query, {"start_date": start_date, "end_date": end_date}).fetchone()
        
        if result and result.total_assessments > 0:
            return {
                "total_assessments": float(result.total_assessments),
                "average_score": float(result.average_score or 0.0),
                "pass_rate": float(result.passed_assessments) / float(result.total_assessments) * 100,
                "average_execution_time": float(result.avg_execution_time or 0.0),
                "unique_pipeline_runs": float(result.unique_runs or 0)
            }
        return {"total_assessments": 0.0, "average_score": 0.0, "pass_rate": 0.0, "average_execution_time": 0.0, "unique_pipeline_runs": 0.0}
    
    def _calculate_gate_performance(self, session: Session, start_date: datetime, end_date: datetime) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each gate type."""
        query = text("""
            SELECT 
                gate_type,
                COUNT(*) as total_assessments,
                AVG(score) as average_score,
                COUNT(*) FILTER (WHERE passed = true) as passed_assessments,
                MIN(score) as min_score,
                MAX(score) as max_score,
                STDDEV(score) as score_stddev
            FROM quality_assessments 
            WHERE created_at >= :start_date AND created_at <= :end_date
            GROUP BY gate_type
        """)
        
        result = session.execute(query, {"start_date": start_date, "end_date": end_date})
        
        gate_performance = {}
        for row in result:
            gate_performance[row.gate_type] = {
                "total_assessments": float(row.total_assessments),
                "average_score": float(row.average_score or 0.0),
                "pass_rate": float(row.passed_assessments) / float(row.total_assessments) * 100,
                "min_score": float(row.min_score or 0.0),
                "max_score": float(row.max_score or 0.0),
                "score_variance": float(row.score_stddev or 0.0) ** 2
            }
        
        return gate_performance
    
    def _calculate_quality_trends(self, session: Session, start_date: datetime, end_date: datetime) -> List[QualityTrend]:
        """Calculate quality trends over time."""
        query = text("""
            SELECT 
                gate_type,
                DATE_TRUNC('week', created_at) as week,
                COUNT(*) as total_assessments,
                AVG(score) as average_score,
                COUNT(*) FILTER (WHERE passed = true) as passed_assessments
            FROM quality_assessments 
            WHERE created_at >= :start_date AND created_at <= :end_date
            GROUP BY gate_type, DATE_TRUNC('week', created_at)
            ORDER BY gate_type, week DESC
        """)
        
        result = session.execute(query, {"start_date": start_date, "end_date": end_date})
        
        trends = []
        gate_weeks = {}
        
        for row in result:
            gate_type = row.gate_type
            if gate_type not in gate_weeks:
                gate_weeks[gate_type] = []
            
            gate_weeks[gate_type].append({
                "week": row.week,
                "average_score": float(row.average_score or 0.0),
                "pass_rate": float(row.passed_assessments) / float(row.total_assessments) * 100,
                "total_assessments": int(row.total_assessments)
            })
        
        # Calculate improvement rates
        for gate_type, weeks in gate_weeks.items():
            if len(weeks) >= 2:
                latest = weeks[0]
                previous = weeks[1]
                improvement_rate = ((latest["average_score"] - previous["average_score"]) / previous["average_score"]) * 100
                
                trends.append(QualityTrend(
                    gate_type=gate_type,
                    period="weekly",
                    average_score=latest["average_score"],
                    pass_rate=latest["pass_rate"],
                    total_assessments=latest["total_assessments"],
                    improvement_rate=improvement_rate
                ))
        
        return trends
    
    def _generate_quality_insights(self, session: Session, start_date: datetime, end_date: datetime) -> QualityInsights:
        """Generate actionable quality insights."""
        query = text("""
            SELECT 
                gate_type,
                AVG(score) as average_score,
                COUNT(*) FILTER (WHERE passed = true) as passed_count,
                COUNT(*) as total_count
            FROM quality_assessments 
            WHERE created_at >= :start_date AND created_at <= :end_date
            GROUP BY gate_type
            ORDER BY average_score ASC
        """)
        
        result = session.execute(query, {"start_date": start_date, "end_date": end_date})
        gate_scores = list(result)
        
        if not gate_scores:
            return QualityInsights([], [], [], 0.0, 1.0)
        
        # Identify weakest and strongest gates
        weakest_gates = [row.gate_type for row in gate_scores[:2]]  # Bottom 2
        strongest_gates = [row.gate_type for row in gate_scores[-2:]]  # Top 2
        
        # Generate improvement opportunities
        improvement_opportunities = []
        for row in gate_scores:
            if row.average_score < 0.85:  # Below threshold
                improvement_opportunities.append(f"Improve {row.gate_type} (avg score: {row.average_score:.2f})")
        
        # Get revision analytics for insights
        revision_analytics = self.get_revision_analytics(days=(end_date - start_date).days)
        
        return QualityInsights(
            weakest_gates=weakest_gates,
            strongest_gates=strongest_gates,
            improvement_opportunities=improvement_opportunities,
            revision_success_rate=revision_analytics.get("revision_success_rate", 0.0),
            average_attempts_to_pass=revision_analytics.get("average_attempts_to_pass", 1.0)
        )
    
    def _analyze_revision_patterns(self, session: Session, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze revision patterns and effectiveness."""
        # This would be enhanced with actual revision tracking data
        # For now, provide basic structure
        
        return {
            "revision_effectiveness": {
                "fact_check_improvement_rate": 85.0,  # Placeholder - would calculate from actual data
                "domain_expertise_improvement_rate": 78.0,
                "style_consistency_improvement_rate": 82.0,
                "compliance_improvement_rate": 91.0
            },
            "common_revision_triggers": [
                "Low citation density",
                "Insufficient domain terminology",
                "Poor readability scores",
                "Missing compliance disclaimers"
            ],
            "revision_success_patterns": {
                "first_revision_success_rate": 72.0,
                "second_revision_success_rate": 89.0,
                "max_revisions_needed": 2
            }
        }
    
    def _get_default_metrics(self) -> QualityMetrics:
        """Return default metrics when analysis fails."""
        return QualityMetrics(
            overall_metrics={"total_assessments": 0.0, "average_score": 0.0, "pass_rate": 0.0},
            gate_performance={},
            trends=[],
            insights=QualityInsights([], [], [], 0.0, 1.0),
            revision_patterns={}
        )

# Global service instance
quality_analytics_service = QualityAnalyticsService()