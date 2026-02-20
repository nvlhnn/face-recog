-- Face Recognition Database Schema (Universal)
-- ==========================================
-- This schema supports multiple engines (OpenCV, InsightFace)
-- and can be used with MySQL, MariaDB, or PostgreSQL.
-- Note: SQLite schema is handled automatically by the application.

-- 1. Users Table
-- Stores face encodings per user AND per engine
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(100) NOT NULL,
    engine VARCHAR(50) NOT NULL,
    encoding LONGTEXT NOT NULL, -- JSON array of floats (128 for OpenCV, 512 for InsightFace)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, engine)
);

-- 2. Indexes for performance
CREATE INDEX idx_users_engine ON users(engine);
CREATE INDEX idx_users_user_id ON users(user_id);
