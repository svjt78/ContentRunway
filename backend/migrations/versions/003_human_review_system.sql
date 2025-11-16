-- Human Review System Redesign
-- Migration: 003_human_review_system.sql
-- Purpose: Add review status fields to content_drafts and prepare for Redis session removal

BEGIN;

-- Add new review fields to content_drafts table
ALTER TABLE content_drafts 
ADD COLUMN IF NOT EXISTS review_status VARCHAR(20) NOT NULL DEFAULT 'draft',
ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP WITH TIME ZONE NULL,
ADD COLUMN IF NOT EXISTS review_notes TEXT NULL,
ADD COLUMN IF NOT EXISTS reviewer_id VARCHAR(255) NULL,
ADD COLUMN IF NOT EXISTS published_at TIMESTAMP WITH TIME ZONE NULL,
ADD COLUMN IF NOT EXISTS published_urls JSON NULL;

-- Create indexes for review status queries
CREATE INDEX IF NOT EXISTS idx_content_drafts_review_status ON content_drafts(review_status);
CREATE INDEX IF NOT EXISTS idx_content_drafts_pipeline_run_id ON content_drafts(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_content_drafts_reviewed_at ON content_drafts(reviewed_at DESC);

-- Migrate existing content to have appropriate status
-- Content from completed pipelines should be marked as 'published'
UPDATE content_drafts SET review_status = 'published', published_at = NOW()
WHERE pipeline_run_id IN (
    SELECT id FROM pipeline_runs WHERE status = 'completed'
);

-- Archive old review session data (for rollback capability)
ALTER TABLE pipeline_runs 
ADD COLUMN IF NOT EXISTS legacy_review_session_id UUID NULL;

-- Copy existing review_session_id to legacy field (if column exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'pipeline_runs' AND column_name = 'review_session_id') THEN
        UPDATE pipeline_runs 
        SET legacy_review_session_id = review_session_id 
        WHERE review_session_id IS NOT NULL;
    END IF;
END $$;

-- Add comment to review_status field for documentation
COMMENT ON COLUMN content_drafts.review_status IS 'Review status: draft (pending), approved (ready for publishing), rejected (needs changes), published (completed)';

COMMIT;