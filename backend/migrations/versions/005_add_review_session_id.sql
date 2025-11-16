-- Add missing review_session_id column to pipeline_runs table
-- Migration: 005_add_review_session_id.sql
-- Purpose: Fix asyncpg.exceptions.UndefinedColumnError for pipeline_runs.review_session_id

BEGIN;

-- Add the missing review_session_id column to pipeline_runs table
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS review_session_id UUID;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_review_session_id ON pipeline_runs(review_session_id);

-- Add comment to document the column purpose
COMMENT ON COLUMN pipeline_runs.review_session_id IS 'Human review session ID for tracking content review process';

COMMIT;