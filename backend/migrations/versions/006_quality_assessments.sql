-- Quality assessments table migration
-- Adds quality_assessments and fact_check_reports tables for tracking quality gate results

-- Ensure uuid-ossp extension exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Quality assessments table
CREATE TABLE IF NOT EXISTS quality_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    content_draft_id UUID REFERENCES content_drafts(id) ON DELETE CASCADE,
    gate_type VARCHAR(50) NOT NULL, -- fact_check, domain_expertise, style_consistency, compliance
    gate_name VARCHAR(100) NOT NULL,
    score DECIMAL(5,4) NOT NULL CHECK (score >= 0 AND score <= 1),
    threshold DECIMAL(5,4) NOT NULL CHECK (threshold >= 0 AND threshold <= 1),
    passed BOOLEAN NOT NULL,
    evidence JSONB DEFAULT '[]',
    suggestions JSONB DEFAULT '[]',
    execution_time_seconds DECIMAL(8,3),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fact check reports table (detailed fact checking results)
CREATE TABLE IF NOT EXISTS fact_check_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    quality_assessment_id UUID REFERENCES quality_assessments(id) ON DELETE CASCADE,
    overall_accuracy_score DECIMAL(5,4) NOT NULL CHECK (overall_accuracy_score >= 0 AND overall_accuracy_score <= 1),
    total_claims INTEGER NOT NULL DEFAULT 0,
    verified_claims INTEGER NOT NULL DEFAULT 0,
    false_claims INTEGER NOT NULL DEFAULT 0,
    unverifiable_claims INTEGER NOT NULL DEFAULT 0,
    claims_analysis JSONB DEFAULT '[]',
    sources_credibility JSONB DEFAULT '{}',
    methodology_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_quality_assessments_pipeline_run ON quality_assessments(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_quality_assessments_gate_type ON quality_assessments(gate_type);
CREATE INDEX IF NOT EXISTS idx_quality_assessments_score ON quality_assessments(score DESC);
CREATE INDEX IF NOT EXISTS idx_quality_assessments_passed ON quality_assessments(passed);

CREATE INDEX IF NOT EXISTS idx_fact_check_reports_pipeline_run ON fact_check_reports(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_fact_check_reports_assessment ON fact_check_reports(quality_assessment_id);
CREATE INDEX IF NOT EXISTS idx_fact_check_reports_accuracy ON fact_check_reports(overall_accuracy_score DESC);