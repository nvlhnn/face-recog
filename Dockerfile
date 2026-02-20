# --- STAGE 1: BUILDER ---
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages and aggressively remove cache/bytecode
RUN pip install --no-cache-dir --user -r requirements.txt gunicorn \
    && find /root/.local -name "*.pyc" -delete \
    && find /root/.local -name "__pycache__" -delete


# --- STAGE 2: RUNTIME ---
FROM python:3.11-slim

WORKDIR /app

# Copy only the compiled Python packages from stage 1
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy only the download script first to bake models in a separate layer
COPY download_models.py .

# Bake models into the image (Download ALL engines by default)
RUN python download_models.py --all

# Copy the rest of the app
COPY . .

# Runtime defaults
ENV ENV=production
ENV PORT=5000
ENV FACE_ENGINE=insightface
ENV INSIGHTFACE_MODEL=buffalo_l

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Workers: override with GUNICORN_WORKERS env var
# For 2GB RAM + InsightFace: use 1 worker
# For 2GB RAM + OpenCV: use 2-3 workers
# --preload shares the model across workers (saves ~40% RAM)
CMD ["sh", "-c", "gunicorn -w ${GUNICORN_WORKERS:-1} --preload -b 0.0.0.0:5000 --timeout 120 run:app"]
