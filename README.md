Sure — here’s a clean, developer-friendly **README.md** you can drop directly into your project.
It explains **what the project does, how to install, how to run registration and real-time attendance**, MongoDB setup, folder structure, and troubleshooting.

If you want it customized (screenshots, diagrams, deployment steps), tell me.

---

# ✅ **README.md**

```md
# Face Recognition Attendance System (MongoDB + Dlib)

This project contains two main modules:

1. **Student Registration (`registration_clean.py`)**  
   Captures face images in 5 head postures (front, left, right, up, down), generates embeddings,
   computes PCA visualization, stores images + embeddings in **MongoDB Atlas (GridFS)**.

2. **Real-time Attendance (`real_time_attendance.py`)**  
   Performs face recognition live using webcam, detects entry/exit, and logs attendance with timestamps.

Both modules use **Dlib** for face alignment & 128-D embeddings, and **MongoDB Atlas** as backend.

---

## ✨ Features

### 🔵 Student Registration
- Captures **5 facial postures**
- Computes **128-D embeddings**
- Saves:
  - mean embedding  
  - posture embeddings  
  - captured images (GridFS)  
  - PCA 3D scatter plot  
- Stores data in MongoDB Atlas under `users` collection
- Voice guidance (optional)

### 🟢 Real-Time Attendance
- Detects faces and matches embeddings
- Shows only:
  - **Name**
  - **Enrollment Number**
  - **Current Time**
- Logs:
  - entry time  
  - exit time  
  - active session status  
- Updates `attendance` collection in MongoDB

---

## 📁 Project Structure

```

project/
│
├── new_verify.py    # Attendance tracking (live camera)
├── new.py      # Student registration with 5 postures
├── models/                    # Dlib model files (auto-download)
│     ├── shape_predictor_68_face_landmarks.dat
│     └── dlib_face_recognition_resnet_model_v1.dat
│
└── README.md

````

---

## ⚙️ Installation

### 1️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Typical required libraries:

```
opencv-python
dlib
numpy
pymongo
tqdm
matplotlib
scikit-learn
requests
```

You also need **CMake** and **Visual Studio Build Tools** on Windows for dlib.

---

## 🛢️ MongoDB Atlas Setup

Create a new database on **MongoDB Atlas** and replace credentials:

```
MONGO_USER
MONGO_PASS
MONGO_HOST
MONGO_DBNAME
```

Either add them to environment variables:

```bash
export MONGO_USER="yourUser"
export MONGO_PASS="yourPassword"
export MONGO_HOST="cluster0.mongodb.net"
export MONGO_DBNAME="elearning_platform"
```

Or keep the default values inside the script (not recommended for production).

---

## 🧑‍🎓 Registering a Student

Run:

```bash
python registration_clean.py
```

You will be prompted:

```
Enter Name:
Enter Enrollment No:
Enter Gmail (@gmail.com):
```

Then the webcam will open and capture:

* front
* left
* right
* up
* down

After successful registration:

✔ Embeddings stored
✔ Images stored
✔ PCA plot stored
✔ User document created/updated

---

## 🟩 Running Real-Time Attendance

Run:

```bash
python real_time_attendance.py
```

The webcam window will display:

```
Name | Enrollment | Time
```

Example:

```
Muskan Rohe | ENR12345 | 10:35:22
```

MongoDB `attendance` collection will store:

```json
{
  "enrollment": "ENR12345",
  "name": "Muskan Rohe",
  "entry_time": "...",
  "exit_time": "...",
  "active": false
}
```

---

## 🔍 Troubleshooting

### ❓ "?" Appears Instead of Enrollment

Fix: ensure enrollment is stored as string.

In `prepare_users()`:

```python
"enrollment": str(d.get("enrollment", ""))
```

---

### ⚠️ "Model files missing"

Run registration first — it auto-downloads Dlib models.

---

### ❌ Webcam Not Found

Try changing camera index:

```python
cap = cv2.VideoCapture(1)
```

---

### ⚙ MongoDB Connection Errors

Check:

* IP whitelist in Atlas
* Correct password (URL-encoded)
* Proper network connectivity

---

## 📜 License

You can add your own license text or leave it open.

---

## 🤝 Contribution

Open to improvements — PRs welcome!

```

---

If you want a **requirements.txt**, **architecture diagram**, or **Flowchart (PlantUML)**, I can add that too.
```
