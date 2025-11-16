-- Migration: Add target_character_count to pipeline_runs
-- Date: 2025-11-13
-- Description: Add configurable content length target to pipeline runs

-- Add target_character_count column with default value of 500
ALTER TABLE pipeline_runs 
ADD COLUMN target_character_count INTEGER NOT NULL DEFAULT 500;

-- Add check constraint to ensure reasonable values (100-4000 characters)
ALTER TABLE pipeline_runs 
ADD CONSTRAINT pipeline_runs_target_character_count_check 
CHECK (target_character_count >= 100 AND target_character_count <= 4000);

-- Update existing records to have the default value (should be automatic with DEFAULT)
-- This is just for safety in case there are any existing records without the default
UPDATE pipeline_runs 
SET target_character_count = 500 
WHERE target_character_count IS NULL;

-- Add index for performance if querying by target length
CREATE INDEX idx_pipeline_runs_target_character_count 
ON pipeline_runs (target_character_count);

-- Comment for documentation
COMMENT ON COLUMN pipeline_runs.target_character_count IS 'Target content length in characters (100-4000)';