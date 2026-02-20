# 🚀 Deployment Guide - Face Recognition API

## Target VPS:
- **Spec**: 2 Core CPU, 2GB RAM, 40GB SSD
- **Region**: Jakarta
- **OS**: Ubuntu 22.04 LTS

---

## Step 0: Prepare VPS (First-time Setup)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Add 2GB swap (IMPORTANT for 2GB RAM!)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap
free -h

# Install Docker (if not present)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Install Docker Compose plugin
sudo apt install -y docker-compose-plugin

# Log out and back in for group changes
exit
```

---

## Step 1: Upload Code to VPS

### Option A: Using Git (Recommended)
```bash
cd /opt
sudo git clone https://github.com/YOUR_USERNAME/face-recog.git
cd face-recog
sudo chown -R $USER:$USER /opt/face-recog
```

### Option B: Using SCP (from your Windows PC)
```powershell
# On Windows PowerShell
scp -r . ubuntu@YOUR_VPS_IP:/opt/face-recog
```

---

## Step 2: Configure Environment

```bash
cd /opt/face-recog

# Edit production config
cp .env.production .env.production.local
nano .env.production

# IMPORTANT: Change these values!
# ACCESS_TOKEN=your-secure-random-token-here
# FACE_ENGINE=insightface  (or opencv for lower RAM)
```

---

## Step 3: Deploy with Docker

### Option A: Docker Compose (Simple)
```bash
# Build and run
docker compose up --build -d

# Check logs
docker compose logs -f

# Check memory usage
docker stats
```

### Option B: Docker Compose with Traefik (Production)
```bash
# Make sure Traefik network exists
docker network create reverse-proxy

# Deploy
docker compose -f docker-compose.yml up --build -d
```

---

## Step 4: Verify Deployment

```bash
# Health check
curl http://localhost:5000/health

# Check API docs
# http://YOUR_VPS_IP:5000/apidocs/

# Test liveness
curl -X POST http://localhost:5000/liveness/start

# Check memory
docker stats --no-stream
free -h
```

---

## 📊 Expected Resource Usage (2GB VPS)

| Engine | Workers | RAM (idle) | RAM (peak) | Throughput |
|--------|---------|------------|------------|------------|
| InsightFace | 1 | ~600MB | ~800MB | ~4-6 req/s |
| OpenCV | 2 | ~400MB | ~600MB | ~6-10 req/s |

> With 2GB swap enabled, peak memory spikes won't crash the app.

---

## 🔧 Tuning for 2GB RAM

### If using InsightFace (default):
```bash
# In docker-compose.yml or .env.production:
GUNICORN_WORKERS=1    # Max 1 worker with InsightFace on 2GB
```

### If using OpenCV (lighter):
```bash
FACE_ENGINE=opencv
GUNICORN_WORKERS=2    # Can run 2 workers with OpenCV
```

### Switch dynamically:
```bash
# Edit .env.production
nano .env.production
# Change FACE_ENGINE and GUNICORN_WORKERS

# Restart
docker compose restart
```

---

## 🔧 Useful Commands

```bash
# Restart service
docker compose restart

# Stop service
docker compose down

# Rebuild after code changes
docker compose up --build -d

# View logs (follow)
docker compose logs -f

# View logs (last 100 lines)
docker compose logs --tail=100

# Check memory usage
docker stats --no-stream
free -h

# Enter container for debugging
docker exec -it api-face-recog bash

# Cleanup old images (free disk space)
docker system prune -af
```

---

## 🔐 Firewall

```bash
# Open port 5000 (direct access)
sudo ufw allow 5000/tcp

# Or if using Traefik, only open 80/443
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

---

## ⚠️ Important Notes for 2GB VPS

1. **Swap is essential** - Without swap, InsightFace model loading can OOM-kill the container
2. **1 worker only** for InsightFace - each worker loads ~500MB model
3. **`--preload` flag** in Gunicorn shares model memory between workers (already configured)
4. **Monitor with** `docker stats` and `free -h` regularly
5. **0.5TB traffic** = ~500GB/mo. Each liveness request is ~100KB-1MB, so this supports ~500K-5M requests/mo easily
