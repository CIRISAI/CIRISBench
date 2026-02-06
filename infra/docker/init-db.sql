-- Initialize CIRISBench Database
-- This runs on first PostgreSQL container startup

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =============================================================================
-- Unified Evaluation Pipeline tables (FSD: Unified Evaluation Pipeline)
-- =============================================================================

-- evaluations: single source of truth for all HE-300 evaluation results
CREATE TABLE IF NOT EXISTS evaluations (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       VARCHAR(64) NOT NULL,
    eval_type       VARCHAR(16) NOT NULL,
    target_model    VARCHAR(128),
    target_provider VARCHAR(64),
    target_endpoint TEXT,
    model_version   VARCHAR(64),
    protocol        VARCHAR(16) NOT NULL,
    agent_name      VARCHAR(128),
    sample_size     INT         NOT NULL DEFAULT 300,
    seed            INT         NOT NULL,
    concurrency     INT         NOT NULL DEFAULT 50,
    batch_config    JSONB,
    status          VARCHAR(16) NOT NULL DEFAULT 'queued',
    accuracy        FLOAT,
    total_scenarios INT,
    correct         INT,
    errors          INT,
    categories      JSONB,
    avg_latency_ms  FLOAT,
    processing_ms   INT,
    scenario_results JSONB,
    trace_id        VARCHAR(128),
    trace_binding   JSONB,
    visibility      VARCHAR(8)  NOT NULL DEFAULT 'private',
    badges          JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    CONSTRAINT valid_visibility CHECK (visibility IN ('public', 'private')),
    CONSTRAINT valid_status CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    CONSTRAINT valid_eval_type CHECK (eval_type IN ('frontier', 'client'))
);

-- Partial indexes for query patterns
CREATE INDEX IF NOT EXISTS idx_eval_public
    ON evaluations (visibility, status, accuracy DESC)
    WHERE visibility = 'public' AND status = 'completed';

CREATE INDEX IF NOT EXISTS idx_eval_frontier_model
    ON evaluations (target_model, completed_at DESC)
    WHERE eval_type = 'frontier' AND status = 'completed';

CREATE INDEX IF NOT EXISTS idx_eval_tenant
    ON evaluations (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_eval_model_history
    ON evaluations (target_model, completed_at DESC)
    WHERE visibility = 'public' AND status = 'completed';

-- frontier_models: registry of models to sweep on cron schedule
CREATE TABLE IF NOT EXISTS frontier_models (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        VARCHAR(128) NOT NULL UNIQUE,
    provider        VARCHAR(64) NOT NULL,
    display_name    VARCHAR(128) NOT NULL,
    provider_label  VARCHAR(64),
    active          BOOLEAN     NOT NULL DEFAULT true,
    proxy_route     VARCHAR(256),
    eval_config     JSONB,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Seed initial frontier model registry (FSD Section 10)
INSERT INTO frontier_models (model_id, provider, display_name, provider_label, proxy_route) VALUES
    ('openai/gpt-4o',              'OpenAI',    'GPT-4o',              'OpenAI',    'openai/gpt-4o'),
    ('openai/gpt-4.5-preview',     'OpenAI',    'GPT-4.5',            'OpenAI',    'openai/gpt-4.5-preview'),
    ('openai/o3',                  'OpenAI',    'o3',                  'OpenAI',    'openai/o3'),
    ('openai/o3-mini',             'OpenAI',    'o3-mini',             'OpenAI',    'openai/o3-mini'),
    ('anthropic/claude-4-opus',    'Anthropic', 'Claude 4 Opus',       'Anthropic', 'anthropic/claude-opus-4-0-20250514'),
    ('anthropic/claude-4-sonnet',  'Anthropic', 'Claude 4 Sonnet',     'Anthropic', 'anthropic/claude-sonnet-4-0-20250514'),
    ('anthropic/claude-3.5-haiku', 'Anthropic', 'Claude 3.5 Haiku',    'Anthropic', 'anthropic/claude-3-5-haiku-20241022'),
    ('google/gemini-2.0-pro',     'Google',    'Gemini 2.0 Pro',      'Google',    'openrouter/google/gemini-2.0-pro'),
    ('google/gemini-2.0-flash',   'Google',    'Gemini 2.0 Flash',    'Google',    'openrouter/google/gemini-2.0-flash'),
    ('meta/llama-4-maverick',     'Meta',      'Llama 4 Maverick',    'Meta',      'openrouter/meta-llama/llama-4-maverick'),
    ('meta/llama-4-scout',        'Meta',      'Llama 4 Scout',       'Meta',      'openrouter/meta-llama/llama-4-scout'),
    ('deepseek/deepseek-r1',      'DeepSeek',  'DeepSeek R1',         'DeepSeek',  'openrouter/deepseek/deepseek-r1'),
    ('mistral/mistral-large',     'Mistral',   'Mistral Large',       'Mistral',   'openrouter/mistralai/mistral-large-latest'),
    ('xai/grok-3',                'xAI',       'Grok 3',              'xAI',       'openrouter/xai/grok-3'),
    ('cohere/command-r-plus',     'Cohere',    'Command R+',          'Cohere',    'openrouter/cohere/command-r-plus')
ON CONFLICT (model_id) DO NOTHING;
