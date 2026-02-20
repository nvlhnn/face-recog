# Face Recognition API

A multi-engine face recognition service for face registration, verification, and analysis.

**Architecture:** 1:1 Verification (Stored Encoding vs Live Photo)  
**Storage:** SQLite + Encoding Vectors  

## ⚡ Key Features

- **Multi-Engine Support:** Swap between OpenCV and InsightFace via a single env var
- **Anti-Spoofing:** Built-in liveness detection to prevent photo/screen spoofing
- **Low Resource Usage:** Runs on standard CPU (no GPU needed)
- **Zero License Cost:** Uses MIT/Apache-licensed libraries
- **Fast Verification:** ~100-200ms verification time (OpenCV engine)
- **Attendance Logging:** Automatically records timestamps in SQLite
- **Swagger Docs:** Interactive API documentation at `/apidocs`

## 🔧 Supported Engines

| Engine | Env Value | RAM | Speed | Features |
|---|---|---|---|---|
| **OpenCV** (default) | `opencv` | ~150MB | ~100-200ms | Eyes open, smile detection |
| **InsightFace** | `insightface` | ~500MB | ~200-400ms | Age, gender estimation |

## 🛠️ Setup & Run

### Option 1: Local (Python)

```bash
# 1. Create virtual environment
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# For InsightFace engine (optional):
pip install insightface onnxruntime

# 3. Create env config
cp .env.example .env.staging   # for development
cp .env.example .env.production # for production

# 4. Edit your config
# Set FACE_ENGINE, ACCESS_TOKEN, ANTI_SPOOFING, etc.

# 5. Run (development)
set ENV=staging          # Windows
export ENV=staging       # Linux
python run.py
# → http://localhost:5000

# 5b. Run (production with gunicorn - Linux only)
ENV=production gunicorn -w 2 -b 0.0.0.0:5000 run:app
```

### Option 2: Docker (Local Development)

```bash
# Build and run with exposed port
docker compose -f docker-compose.local.yml up --build

# → http://localhost:5000
# Default token: dev-token-123
```

### Option 3: Docker (Production with Traefik)

```bash
# Set your access token
export ACCESS_TOKEN=your-secure-token

# Deploy
docker compose up -d --build

# → Accessible via Traefik reverse proxy
```

### ⚙️ Configuration

Create `.env.staging` or `.env.production` from `.env.example`:

```bash
# Engine: opencv (default) or insightface
FACE_ENGINE=opencv

# API Security
ACCESS_TOKEN=your-secret-token

# InsightFace options (only when FACE_ENGINE=insightface)
INSIGHTFACE_MODEL=buffalo_l

# Anti-Spoofing (optional)
ANTI_SPOOFING=false
ANTI_SPOOFING_THRESHOLD=0.5
```

> **Note:** Models are downloaded automatically on first run. OpenCV models (~37MB) are downloaded by `download_models.py`. InsightFace models are auto-downloaded by the library.

## 📱 API Usage

### Register Face
```bash
curl -X POST http://localhost:5000/register \
  -F "user_id=EMP001" \
  -F "image=@official_photo.jpg"
```

### Verify Face
```bash
curl -X POST http://localhost:5000/verify \
  -F "user_id=EMP001" \
  -F "image=@selfie.jpg"
```
**Response:**
```json
{
    "status": "success",
    "matched": true,
    "user_id": "EMP001",
    "distance": 0.1234,
    "confidence": 85.2
}
```

### Analyze Face Attributes
```bash
curl -X POST http://localhost:5000/analyze \
  -F "image=@photo.jpg"
```

**Response varies by engine:**

<details>
<summary><b>OpenCV Engine</b></summary>

```json
{
    "status": "success",
    "engine": "opencv",
    "face_detected": true,
    "eyes_open": true,
    "smiling": true,
    "smile_confidence": 82.5,
    "landmarks": {
        "right_eye": [180.2, 150.3],
        "left_eye": [250.1, 148.7],
        "nose": [215.0, 190.5],
        "right_mouth": [190.3, 220.1],
        "left_mouth": [240.8, 219.6]
    }
}
```
</details>

<details>
<summary><b>InsightFace Engine</b></summary>

```json
{
    "status": "success",
    "engine": "insightface",
    "face_detected": true,
    "age": 28,
    "gender": "male",
    "landmarks": {
        "left_eye": [180.2, 150.3],
        "right_eye": [250.1, 148.7],
        "nose": [215.0, 190.5],
        "left_mouth": [190.3, 220.1],
        "right_mouth": [240.8, 219.6]
    }
}
```
</details>

### Liveness Check (Anti-Spoofing)
```bash
curl -X POST http://localhost:5000/liveness \
  -F "image=@selfie.jpg"
```
**Response:**
```json
{
    "status": "success",
    "engine": "opencv",
    "face_detected": true,
    "is_live": true,
    "liveness_score": 0.7823,
    "details": {
        "sharpness": 1.0,
        "color": 0.7,
        "moire": 0.8,
        "edges": 0.6,
        "reflection": 1.0
    },
    "message": null
}
```

> **Tip:** Set `ANTI_SPOOFING=true` to automatically check liveness during `/register` and `/verify`.

## 📂 Project Structure

```
face-recog/
├── app/
│   ├── api/              # REST endpoints
│   ├── engines/          # Face engine abstraction
│   │   ├── base.py       # Abstract interface (FaceEngineBase)
│   │   ├── anti_spoof.py # Texture-based liveness detection
│   │   ├── opencv_engine.py
│   │   └── insightface_engine.py
│   ├── services/         # Business logic
│   ├── repositories/     # Data access (SQLite)
│   └── utils/            # Image processing utilities
├── models/               # OpenCV ONNX models (auto-downloaded)
├── data/                 # SQLite databases
└── requirements.txt
```

## 📊 Database
- `users` — Registered users with face encodings
- `attendance_logs` — History of all verifications
