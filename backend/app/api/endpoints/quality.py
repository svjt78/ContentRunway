from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import uuid

from app.db.database import get_db

router = APIRouter()

@router.get("/assessments/{run_id}")
async def get_quality_assessments(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get quality assessments for a pipeline run."""
    try:
        from sqlalchemy import text
        
        # Get quality assessments - using actual schema fields
        assessment_query = text("""
            SELECT 
                id, pipeline_run_id, assessor_type as gate_type, gate_name, content_draft_id,
                overall_score as score, threshold_used as threshold, passed, evidence, suggestions, 
                processing_time_seconds as execution_time_seconds, created_at
            FROM quality_assessments 
            WHERE pipeline_run_id = :run_id 
            ORDER BY assessor_type, created_at DESC
        """)
        
        result = await db.execute(assessment_query, {"run_id": str(run_id)})
        assessment_rows = result.fetchall()
        
        # Get fact check reports if available - using actual schema fields
        fact_check_query = text("""
            SELECT 
                fcr.id, qa.pipeline_run_id, 
                qa.overall_score as overall_accuracy_score, fcr.total_claims,
                fcr.verified_claims, fcr.disputed_claims as false_claims, 
                fcr.unverifiable_claims, fcr.claims_analysis,
                fcr.supporting_evidence as sources_credibility, 
                'Auto-generated fact check report' as methodology_notes, 
                fcr.created_at
            FROM fact_check_reports fcr
            JOIN quality_assessments qa ON fcr.quality_assessment_id = qa.id
            WHERE qa.pipeline_run_id = :run_id 
            ORDER BY fcr.created_at DESC
        """)
        
        fact_result = await db.execute(fact_check_query, {"run_id": str(run_id)})
        fact_rows = fact_result.fetchall()
        
        # Group assessments by gate type
        assessments_by_gate = {}
        for row in assessment_rows:
            gate_type = row.gate_type
            if gate_type not in assessments_by_gate:
                assessments_by_gate[gate_type] = []
            
            assessment = {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "gate_type": row.gate_type,
                "gate_name": row.gate_name,
                "content_draft_id": str(row.content_draft_id) if row.content_draft_id else None,
                "score": row.score,
                "threshold": row.threshold,
                "passed": row.passed,
                "evidence": row.evidence if row.evidence else [],
                "suggestions": row.suggestions if row.suggestions else [],
                "execution_time_seconds": row.execution_time_seconds,
                "created_at": row.created_at.isoformat() if row.created_at else None
            }
            assessments_by_gate[gate_type].append(assessment)
        
        # Process fact check reports
        fact_check_reports = []
        for row in fact_rows:
            report = {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "overall_accuracy_score": row.overall_accuracy_score,
                "total_claims": row.total_claims,
                "verified_claims": row.verified_claims,
                "false_claims": row.false_claims,
                "unverifiable_claims": row.unverifiable_claims,
                "claims_analysis": row.claims_analysis if row.claims_analysis else [],
                "sources_credibility": row.sources_credibility if row.sources_credibility else {},
                "methodology_notes": row.methodology_notes,
                "created_at": row.created_at.isoformat() if row.created_at else None
            }
            fact_check_reports.append(report)
        
        # Calculate overall score
        all_scores = [assessment["score"] for assessments in assessments_by_gate.values() for assessment in assessments]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            "pipeline_run_id": str(run_id),
            "assessments_by_gate": assessments_by_gate,
            "fact_check_reports": fact_check_reports,
            "overall_score": overall_score,
            "total_assessments": len(assessment_rows),
            "gates_passed": sum(1 for assessments in assessments_by_gate.values() for assessment in assessments if assessment["passed"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch quality assessments: {str(e)}")

@router.get("/critique/{run_id}")
async def get_critique_reports(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get critique reports for a pipeline run."""
    try:
        from sqlalchemy import text
        
        query = text("""
            SELECT 
                id, pipeline_run_id, content_draft_id, cycle, critique_text,
                recommendations, quality_score, decision, decision_reasoning,
                improvement_suggestions, created_at
            FROM critique_reports 
            WHERE pipeline_run_id = :run_id 
            ORDER BY cycle DESC, created_at DESC
        """)
        
        result = await db.execute(query, {"run_id": str(run_id)})
        rows = result.fetchall()
        
        if not rows:
            return []
        
        reports = []
        for row in rows:
            report = {
                "id": str(row.id),
                "pipeline_run_id": str(row.pipeline_run_id),
                "content_draft_id": str(row.content_draft_id) if row.content_draft_id else None,
                "cycle": row.cycle,
                "critique_text": row.critique_text,
                "recommendations": row.recommendations if row.recommendations else [],
                "quality_score": row.quality_score,
                "decision": row.decision,
                "decision_reasoning": row.decision_reasoning,
                "improvement_suggestions": row.improvement_suggestions if row.improvement_suggestions else [],
                "created_at": row.created_at.isoformat() if row.created_at else None
            }
            reports.append(report)
        
        return reports
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch critique reports: {str(e)}")

@router.get("/scores/{run_id}")
async def get_quality_scores(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get quality scores for a pipeline run."""
    # Placeholder implementation
    return {"message": "Quality scores endpoint - implementation pending"}