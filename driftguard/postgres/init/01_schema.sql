-- ─────────────────────────────────────────
-- DriftGuard Database Schema
-- Auto-runs on first PostgreSQL startup
-- ─────────────────────────────────────────

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────
-- TABLE 1: models
-- Stores every trained model version
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS models (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version         VARCHAR(50) NOT NULL UNIQUE,
    mlflow_run_id   VARCHAR(100),
    trained_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accuracy        FLOAT,
    precision_score FLOAT,
    recall_score    FLOAT,
    f1_score        FLOAT,
    auc_roc         FLOAT,
    status          VARCHAR(20) DEFAULT 'active'
                    CHECK (status IN ('active', 'retired', 'rejected', 'candidate')),
    retrain_reason  TEXT,
    training_samples INTEGER,
    notes           TEXT
);

-- ─────────────────────────────────────────
-- TABLE 2: drift_events
-- Every detected drift is logged here
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS drift_events (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detected_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    batch_date          DATE NOT NULL,
    feature_name        VARCHAR(100),
    drift_type          VARCHAR(20) NOT NULL
                        CHECK (drift_type IN ('data_drift', 'concept_drift', 'both', 'unknown')),
    detection_method    VARCHAR(20)
                        CHECK (detection_method IN ('PSI', 'KS', 'ADWIN', 'KL', 'combined')),
    drift_score         FLOAT,
    threshold_used      FLOAT,
    severity            VARCHAR(10)
                        CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    triggered_retrain   BOOLEAN DEFAULT FALSE,
    batch_size          INTEGER,
    notes               TEXT
);

-- ─────────────────────────────────────────
-- TABLE 3: retraining_log
-- Every retrain attempt, pass or fail
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retraining_log (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    drift_event_id      UUID REFERENCES drift_events(id) ON DELETE SET NULL,
    old_model_id        UUID REFERENCES models(id) ON DELETE SET NULL,
    new_model_id        UUID REFERENCES models(id) ON DELETE SET NULL,
    triggered_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at        TIMESTAMP,
    decision            VARCHAR(20)
                        CHECK (decision IN ('deployed', 'rejected', 'failed', 'pending')),
    old_accuracy        FLOAT,
    new_accuracy        FLOAT,
    improvement         FLOAT,
    training_duration_s INTEGER,
    failure_reason      TEXT
);

-- ─────────────────────────────────────────
-- TABLE 4: predictions
-- Stores predictions for A/B testing
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id        UUID REFERENCES models(id) ON DELETE CASCADE,
    predicted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    batch_date      DATE,
    input_hash      VARCHAR(64),
    prediction      INTEGER CHECK (prediction IN (0, 1)),
    confidence      FLOAT,
    actual_label    INTEGER CHECK (actual_label IN (0, 1)),
    is_ab_test      BOOLEAN DEFAULT FALSE,
    ab_group        VARCHAR(10) CHECK (ab_group IN ('control', 'challenger'))
);

-- ─────────────────────────────────────────
-- TABLE 5: shap_explanations
-- SHAP feature importance per drift event
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS shap_explanations (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    drift_event_id  UUID REFERENCES drift_events(id) ON DELETE CASCADE,
    model_id        UUID REFERENCES models(id) ON DELETE CASCADE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_name    VARCHAR(100) NOT NULL,
    shap_value      FLOAT NOT NULL,
    mean_abs_shap   FLOAT,
    rank            INTEGER,
    direction       VARCHAR(10) CHECK (direction IN ('positive', 'negative'))
);

-- ─────────────────────────────────────────
-- TABLE 6: system_logs
-- General system activity log
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_logs (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level       VARCHAR(10) CHECK (level IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    component   VARCHAR(50),
    message     TEXT,
    metadata    JSONB
);

-- ─────────────────────────────────────────
-- TABLE 7: reference_stats
-- Baseline distribution stats (backup to Redis)
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reference_stats (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_name    VARCHAR(100) NOT NULL,
    stat_type       VARCHAR(20),
    mean_val        FLOAT,
    std_val         FLOAT,
    min_val         FLOAT,
    max_val         FLOAT,
    histogram_json  JSONB,
    is_active       BOOLEAN DEFAULT TRUE
);

-- ─────────────────────────────────────────
-- INDEXES for Performance
-- ─────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_drift_events_detected_at ON drift_events(detected_at);
CREATE INDEX IF NOT EXISTS idx_drift_events_batch_date ON drift_events(batch_date);
CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_batch_date ON predictions(batch_date);
CREATE INDEX IF NOT EXISTS idx_retraining_log_triggered_at ON retraining_log(triggered_at);
CREATE INDEX IF NOT EXISTS idx_shap_drift_event ON shap_explanations(drift_event_id);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);

-- ─────────────────────────────────────────
-- SEED: Insert initial model record
-- ─────────────────────────────────────────
INSERT INTO models (version, status, notes)
VALUES ('v0.0.0-init', 'retired', 'System initialization placeholder')
ON CONFLICT (version) DO NOTHING;

-- Done
SELECT 'DriftGuard schema initialized successfully.' AS status;
