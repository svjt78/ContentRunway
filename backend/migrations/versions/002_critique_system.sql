-- Add critique system tables for progressive agent training
-- Migration: 002_critique_system.sql

-- Critique reports table for storing comprehensive post-editing feedback
CREATE TABLE IF NOT EXISTS critique_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id UUID NOT NULL,
    content_draft_id UUID NOT NULL,
    
    -- Critique cycle tracking
    critique_cycle INTEGER NOT NULL DEFAULT 1, -- 1 or 2 (max 2 cycles)
    cycle_type VARCHAR(50) NOT NULL DEFAULT 'initial', -- initial, retry_1, retry_2
    
    -- Quality score progression
    overall_critique_score FLOAT NOT NULL,
    pre_edit_quality_scores JSONB NOT NULL, -- Quality scores before editing
    post_edit_quality_scores JSONB NOT NULL, -- Quality scores after editing
    improvement_effectiveness FLOAT NOT NULL, -- How well editing addressed issues (0.0-1.0)
    
    -- Detailed critique feedback (structured for agent training)
    critique_feedback JSONB NOT NULL, -- Comprehensive feedback structure
    issues_identified JSONB NOT NULL, -- Specific issues found
    issues_resolved JSONB NOT NULL, -- Issues successfully resolved
    issues_remaining JSONB NOT NULL, -- Issues still needing attention
    
    -- Agent performance tracking
    editing_effectiveness JSONB NOT NULL, -- How well editing agent performed
    quality_gate_accuracy JSONB NOT NULL, -- How accurate quality gates were
    
    -- Decision tracking
    retry_decision VARCHAR(20) NOT NULL, -- pass, retry, fail
    retry_reasoning TEXT, -- Why this decision was made
    next_action_required VARCHAR(100), -- What should happen next
    
    -- Processing metadata
    processing_time_seconds FLOAT,
    model_used VARCHAR(100),
    cost_estimate FLOAT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent improvement metrics for progressive training
CREATE TABLE IF NOT EXISTS agent_improvement_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id UUID NOT NULL,
    
    -- Agent identification
    agent_name VARCHAR(100) NOT NULL, -- research, writing, editing, quality_gates, etc.
    agent_version VARCHAR(50) DEFAULT '1.0',
    
    -- Performance tracking
    performance_before JSONB NOT NULL, -- Agent output quality before critique
    performance_after JSONB NOT NULL, -- Agent output quality after critique feedback
    performance_delta FLOAT NOT NULL, -- Change in performance (-1.0 to 1.0)
    
    -- Improvement analysis
    improvement_areas JSONB NOT NULL, -- Specific areas where agent improved/failed
    successful_strategies JSONB NOT NULL, -- Strategies that worked well
    failed_strategies JSONB NOT NULL, -- Strategies that didn't work
    
    -- Feedback application tracking
    feedback_received JSONB NOT NULL, -- Critique feedback given to agent
    feedback_applied JSONB NOT NULL, -- Which feedback was successfully applied
    feedback_ignored JSONB NOT NULL, -- Which feedback was not applied
    
    -- Success indicators
    success_indicators JSONB NOT NULL, -- What made this cycle successful/unsuccessful
    pattern_recognition JSONB, -- Patterns identified for future improvement
    
    -- Training data preparation
    training_features JSONB, -- Features for ML training
    training_labels JSONB, -- Labels for ML training
    data_quality_score FLOAT DEFAULT 1.0, -- Quality of this training sample
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Critique feedback history for tracking improvement progression
CREATE TABLE IF NOT EXISTS critique_feedback_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id UUID NOT NULL,
    content_draft_id UUID NOT NULL,
    
    -- Feedback tracking
    feedback_cycle INTEGER NOT NULL,
    feedback_type VARCHAR(50) NOT NULL, -- comprehensive, focused, final
    
    -- Feedback content
    feedback_summary TEXT NOT NULL,
    specific_feedback JSONB NOT NULL, -- Detailed structured feedback
    priority_areas JSONB NOT NULL, -- Areas needing immediate attention
    
    -- Progress tracking
    previous_issues JSONB, -- Issues from previous cycle
    resolved_issues JSONB NOT NULL, -- Issues resolved in this cycle
    new_issues JSONB NOT NULL, -- New issues discovered
    persistent_issues JSONB NOT NULL, -- Issues that remain unresolved
    
    -- Quality progression
    quality_trend VARCHAR(20) NOT NULL, -- improving, declining, stable
    score_progression JSONB NOT NULL, -- Score changes over cycles
    
    -- Learning data
    effective_feedback JSONB, -- Feedback that led to improvements
    ineffective_feedback JSONB, -- Feedback that didn't help
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance analytics view for agent training
CREATE OR REPLACE VIEW agent_performance_analytics AS
SELECT 
    aim.agent_name,
    COUNT(*) as total_runs,
    AVG(aim.performance_delta) as avg_performance_change,
    AVG(cr.improvement_effectiveness) as avg_improvement_effectiveness,
    COUNT(CASE WHEN cr.retry_decision = 'pass' THEN 1 END) as successful_critiques,
    COUNT(CASE WHEN cr.retry_decision = 'retry' THEN 1 END) as retry_critiques,
    COUNT(CASE WHEN cr.retry_decision = 'fail' THEN 1 END) as failed_critiques,
    DATE_TRUNC('week', aim.created_at) as week
FROM agent_improvement_metrics aim
JOIN critique_reports cr ON aim.pipeline_run_id = cr.pipeline_run_id
GROUP BY aim.agent_name, DATE_TRUNC('week', aim.created_at)
ORDER BY week DESC, avg_performance_change DESC;

-- Quality improvement tracking view
CREATE OR REPLACE VIEW quality_improvement_trends AS
SELECT 
    cr.critique_cycle,
    AVG(cr.improvement_effectiveness) as avg_improvement,
    AVG(cr.overall_critique_score) as avg_final_score,
    COUNT(*) as total_cycles,
    DATE_TRUNC('day', cr.created_at) as date
FROM critique_reports cr
GROUP BY cr.critique_cycle, DATE_TRUNC('day', cr.created_at)
ORDER BY date DESC, cr.critique_cycle;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_critique_reports_pipeline_run ON critique_reports(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_critique_reports_cycle ON critique_reports(critique_cycle);
CREATE INDEX IF NOT EXISTS idx_critique_reports_decision ON critique_reports(retry_decision);
CREATE INDEX IF NOT EXISTS idx_critique_reports_created_at ON critique_reports(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_improvement_agent_name ON agent_improvement_metrics(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_improvement_pipeline_run ON agent_improvement_metrics(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_agent_improvement_performance ON agent_improvement_metrics(performance_delta DESC);
CREATE INDEX IF NOT EXISTS idx_agent_improvement_created_at ON agent_improvement_metrics(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_critique_feedback_pipeline_run ON critique_feedback_history(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_critique_feedback_cycle ON critique_feedback_history(feedback_cycle);
CREATE INDEX IF NOT EXISTS idx_critique_feedback_created_at ON critique_feedback_history(created_at DESC);

-- Add critique tracking fields to existing content_drafts table
ALTER TABLE content_drafts 
ADD COLUMN IF NOT EXISTS critique_cycle_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS final_critique_score FLOAT,
ADD COLUMN IF NOT EXISTS critique_feedback_summary JSONB;

-- Add agent performance tracking to existing pipeline_runs table
ALTER TABLE pipeline_runs 
ADD COLUMN IF NOT EXISTS total_critique_cycles INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS critique_success_rate FLOAT,
ADD COLUMN IF NOT EXISTS agent_performance_trends JSONB,
ADD COLUMN IF NOT EXISTS learning_data_quality FLOAT DEFAULT 1.0;