# Professional Attendance System (Budget Edition)

A cost-effective, high-performance face attendance system designed for local deployment.

**Architecture:** 1:1 Verification (Storage vs Live Photo)
**Engine:** DeepFace (SFace Model) + TensorFlow CPU
**Storage:** SQLite + Local File System

## ⚡ Key Features for 1,000 Users

*   **Low Resource Usage:** Runs on standard CPU (no GPU needed).
*   **Zero License Cost:** Uses MIT-licensed libraries.
*   **Fast Verification:** ~200ms verification time.
*   **Attendance Logging:** Automatically records timestamps in SQLite.

## 🛠️ Setup (No Docker)

### 1. Create Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Server
```bash
python run.py
```
Server will start at `http://0.0.0.0:5000`

## 📱 API Usage

### Clock In (Verify)
Send the User ID and a live selfie.
```bash
curl -X POST http://localhost:5000/verify \
  -F "user_id=EMP001" \
  -F "image=@selfie.jpg"
```
**Response:**
```json
{
    "matched": true,
    "user_id": "EMP001",
    "status": "success"
}
```
*Successfully verified requests are automatically logged to the `attendance_logs` table.*

### Register New Employee
```bash
curl -X POST http://localhost:5000/register \
  -F "user_id=EMP001" \
  -F "image=@official_photo.jpg"
```

## 📂 Data Location
*   **Photos:** `data/faces/EMP001.jpg`
*   **Database:** `data/face_recognition.db`

## 📊 Database Schema
You can open `data/face_recognition.db` with any SQLite viewer.
*   `users`: Basic employee info.
*   `attendance_logs`: History of all clock-ins.
