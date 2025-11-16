-- Migration: Convert target_character_count to target_word_count
-- Date: 2025-11-13
-- Description: Convert content length from character-based to word-based targeting

-- Rename column from target_character_count to target_word_count
ALTER TABLE pipeline_runs 
RENAME COLUMN target_character_count TO target_word_count;

-- Convert existing character counts to word counts (divide by 5, round up)
-- Average English word is approximately 5 characters
UPDATE pipeline_runs 
SET target_word_count = CEIL(target_word_count::float / 5);

-- Drop old constraint
ALTER TABLE pipeline_runs 
DROP CONSTRAINT pipeline_runs_target_character_count_check;

-- Add new constraint for word range (50-2000 words)
ALTER TABLE pipeline_runs 
ADD CONSTRAINT pipeline_runs_target_word_count_check 
CHECK (target_word_count >= 50 AND target_word_count <= 2000);

-- Drop old index
DROP INDEX idx_pipeline_runs_target_character_count;

-- Create new index with updated name
CREATE INDEX idx_pipeline_runs_target_word_count 
ON pipeline_runs (target_word_count);

-- Update column comment
COMMENT ON COLUMN pipeline_runs.target_word_count IS 'Target content length in words (50-2000)';

-- Display converted values for verification
SELECT id, target_word_count, created_at 
FROM pipeline_runs 
ORDER BY created_at DESC 
LIMIT 5;