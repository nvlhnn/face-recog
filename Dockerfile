# Lightweight Face Recognition API
# Using Python 3.11 slim image (~150MB)

FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY . .

# Download models during build (faster startup)
RUN python download_models.py

# Create data directory
RUN mkdir -p data

# Environment variables
ENV ENV=production
ENV PORT=5000

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run with Gunicorn (production server)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "run:app"]
