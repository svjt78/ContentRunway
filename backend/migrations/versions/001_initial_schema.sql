-- Initial database schema for ContentRunway
-- This will be automatically run when the PostgreSQL container starts

-- Create database if not exists (handled by Docker environment)

-- Create tables for the ContentRunway application
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Pipeline runs table
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(255) NOT NULL DEFAULT 'personal',
    status VARCHAR(50) NOT NULL DEFAULT 'initialized',
    domain_focus JSONB NOT NULL,
    quality_thresholds JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    current_step VARCHAR(100),
    progress_percentage FLOAT DEFAULT 0.0,
    chosen_topic_id UUID,
    final_quality_score FLOAT,
    human_approved BOOLEAN DEFAULT FALSE,
    published_urls JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    checkpoint_data JSONB
);

-- Topic ideas table
CREATE TABLE IF NOT EXISTS topic_ideas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id UUID NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    domain VARCHAR(100) NOT NULL,
    relevance_score FLOAT NOT NULL,
    novelty_score FLOAT NOT NULL,
    seo_difficulty FLOAT NOT NULL,
    overall_score FLOAT NOT NULL,
    target_keywords JSONB NOT NULL,
    estimated_traffic INTEGER,
    competition_level VARCHAR(20),
    source_count INTEGER DEFAULT 0,
    sources JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_selected BOOLEAN DEFAULT FALSE
);

-- Research sources table
CREATE TABLE IF NOT EXISTS research_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id UUID NOT NULL,
    topic_id UUID,
    url VARCHAR(1000) NOT NULL,
    title VARCHAR(500) NOT NULL,
    author VARCHAR(200),
    publication_date TIMESTAMPTZ,
    domain VARCHAR(100) NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    summary TEXT NOT NULL,
    key_points JSONB NOT NULL,
    quotable_content JSONB,
    credibility_score FLOAT NOT NULL,
    relevance_score FLOAT NOT NULL,
    currency_score FLOAT NOT NULL,
    citation_count INTEGER DEFAULT 0,
    used_in_content BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Content drafts table
CREATE TABLE IF NOT EXISTS content_drafts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id UUID NOT NULL,
    topic_id UUID NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    stage VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    subtitle VARCHAR(500),
    abstract TEXT,
    outline JSONB,
    content TEXT NOT NULL,
    citations JSONB NOT NULL,
    internal_links JSONB,
    word_count INTEGER NOT NULL,
    reading_time_minutes INTEGER NOT NULL,
    readability_score FLOAT,
    meta_description TEXT,
    keywords JSONB NOT NULL,
    tags JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_current BOOLEAN DEFAULT TRUE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_tenant_status ON pipeline_runs(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_created_at ON pipeline_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_topic_ideas_pipeline_run ON topic_ideas(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_topic_ideas_score ON topic_ideas(overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_research_sources_pipeline_run ON research_sources(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_research_sources_topic ON research_sources(topic_id);
CREATE INDEX IF NOT EXISTS idx_content_drafts_pipeline_run ON content_drafts(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_content_drafts_current ON content_drafts(pipeline_run_id, is_current) WHERE is_current = TRUE;