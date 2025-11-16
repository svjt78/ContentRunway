-- Agent logs table migration
-- Adds agent_logs table for tracking agent activity during pipeline execution

-- Ensure uuid-ossp extension exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Agent logs table
CREATE TABLE IF NOT EXISTS agent_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    agent_name VARCHAR(100) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    context JSONB,
    level VARCHAR(20) NOT NULL DEFAULT 'INFO', -- INFO, WARNING, ERROR, DEBUG
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_logs_pipeline_run ON agent_logs(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_timestamp ON agent_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_logs_level ON agent_logs(level);
CREATE INDEX IF NOT EXISTS idx_agent_logs_agent_name ON agent_logs(agent_name);