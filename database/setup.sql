-- Face Recognition Database Setup
-- ================================
-- Run this script to create the necessary database and tables

-- Create Database (if not exists)
CREATE DATABASE IF NOT EXISTS face_recognition_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE face_recognition_db;

-- Create face_encodings table
CREATE TABLE IF NOT EXISTS face_encodings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL UNIQUE,
    encoding LONGTEXT NOT NULL COMMENT 'JSON serialized face encoding array (128 dimensions)',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Show created tables
SHOW TABLES;

-- Describe the main table
DESCRIBE face_encodings;
