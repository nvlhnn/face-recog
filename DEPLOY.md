# 🚀 Deployment Guide - Face Recognition API

## Prerequisites
- Ubuntu VPS (yours: 2 CPU, 4GB RAM ✅)
- SSH access to your server
- Python 3.10+ installed

---

## Step 1: Upload Code to VPS

### Option A: Using Git (Recommended)
```bash
# On your VPS
cd /opt
sudo git clone https://github.com/YOUR_USERNAME/face-recog.git
cd face-recog
```

### Option B: Using SCP (from your Windows PC)
```powershell
# On Windows PowerShell - zip and upload
Compress-Archive -Path . -DestinationPath face-recog.zip
scp face-recog.zip ubuntu@YOUR_VPS_IP:/opt/
```
```bash
# On VPS - unzip
cd /opt
sudo unzip face-recog.zip -d face-recog
cd face-recog
```

---

## Step 2: Install Python & Dependencies

```bash
# Install Python 3.11 and pip
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3: Configure Environment

```bash
# Copy and edit production config
cp .env.production .env.production.local
nano .env.production.local

# Set your production token!
# ACCESS_TOKEN=your-secure-random-token-here
```

---

## Step 4: Test Run

```bash
# Activate venv and run
source venv/bin/activate
ENV=production python run.py

# Test in another terminal
curl http://localhost:5000/health
```

---

## Step 5: Install Gunicorn (Production Server)

```bash
pip install gunicorn

# Test with Gunicorn
ENV=production gunicorn -w 2 -b 0.0.0.0:5000 run:app
```

---

## Step 6: Create Systemd Service (Auto-start)

```bash
sudo nano /etc/systemd/system/face-recog.service
```

Paste this content:
```ini
[Unit]
Description=Face Recognition API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/face-recog
Environment="ENV=production"
ExecStart=/opt/face-recog/venv/bin/gunicorn -w 2 -b 0.0.0.0:5000 run:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable face-recog
sudo systemctl start face-recog
sudo systemctl status face-recog
```

---

## Step 7: Open Firewall (if needed)

```bash
sudo ufw allow 5000/tcp
```

---

## ✅ Verify Deployment

```bash
# Check service status
sudo systemctl status face-recog

# Check logs
sudo journalctl -u face-recog -f

# Test API
curl http://YOUR_VPS_IP:5000/health
```

---

## 🔧 Useful Commands

```bash
# Restart service
sudo systemctl restart face-recog

# Stop service
sudo systemctl stop face-recog

# View logs
sudo journalctl -u face-recog -n 100

# Check memory usage
ps aux | grep gunicorn
```

---

## 📊 Expected Resource Usage

| Metric | Value |
|--------|-------|
| RAM (idle) | ~150MB |
| RAM (peak) | ~300MB |
| CPU | <5% idle |
| Disk | ~100MB |
