"""
Configuration Module
====================
Environment-specific configuration (staging/production)
"""

import os
from dotenv import load_dotenv

# Determine which environment to load
# Use: set ENV=production or set ENV=staging before running
env = os.getenv('ENV', 'staging')  # Default to staging for safety

# Load environment-specific .env file
env_file = f'.env.{env}'
if os.path.exists(env_file):
    load_dotenv(env_file)
    print(f"📁 Loaded config from: {env_file}")
else:
    # Fallback to default .env
    load_dotenv()
    print(f"⚠️ No {env_file} found, using .env")


class Config:
    """Base configuration."""
    
    # Environment
    ENV = env
    
    # Flask
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Server
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # API Security
    ACCESS_TOKEN = os.getenv('ACCESS_TOKEN', None)
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_UPLOAD_SIZE', 10 * 1024 * 1024))
    RATE_LIMIT = os.getenv('RATE_LIMIT', '100 per minute')
    
    # Database (SQLite) - Different per environment!
    DB_PATH = os.getenv('DB_PATH', f'data/{env}.db')
    
    # Face Recognition (OpenCV + SFace)
    FACE_TOLERANCE = float(os.getenv('FACE_TOLERANCE', 0.363))
    
    # Pagination
    ITEMS_PER_PAGE = int(os.getenv('ITEMS_PER_PAGE', 10))
