# registration_clean.py
"""
Student face registration (Windows + MongoDB Atlas)
- Auto-download dlib models if missing
- Capture postures (front, left, right, up, down)
- Store embeddings, images and PCA plots to MongoDB Atlas (GridFS)
- Keeps voice guidance via pyttsx3 (optional; fails silently)
"""

import os
import sys
import bz2
import io
import time
import requests
import traceback
from tqdm import tqdm
from urllib.parse import quote_plus
from datetime import datetime

import cv2
import dlib
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymongo import MongoClient
from gridfs import GridFS

# ---------------- CONFIG (EDIT BEFORE RUN) ----------------
# MongoDB Atlas credentials: either set them as environment variables (preferred)
# or paste your values below (not recommended in shared code).
MONGO_USER = os.getenv("MONGO_USER") or "muskanrohe2005"   # change if needed
MONGO_PASS = os.getenv("MONGO_PASS") or "Harshi@9424"     # change if needed
MONGO_HOST = os.getenv("MONGO_HOST") or "cluster0.uka2xrp.mongodb.net"
MONGO_DBNAME = os.getenv("MONGO_DBNAME") or "elearning_platform"

# Model files (downloaded automatically if missing)
MODEL_DIR = "models"
SHAPE_PRED_FILENAME = "shape_predictor_68_face_landmarks.dat"
FACE_REC_FILENAME = "dlib_face_recognition_resnet_model_v1.dat"

SHAPE_PRED_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
FACE_REC_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"

# Capture configuration
POSTURES = ["front", "left", "right", "up", "down"]
MAX_POSTURE_DURATION_SEC = 30  # seconds to attempt each posture
MIN_SUCCESS_FRAMES = 5  # number of consecutive valid frames for posture
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------------------------------------

def ensure_models():
    """Download and decompress dlib model files if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    def download_and_decompress(bz_url, out_path):
        tmp_bz = out_path + ".bz2"
        if os.path.exists(out_path):
            print(f"[OK] Model already exists: {out_path}")
            return

        print(f"[INFO] Downloading {bz_url} ...")
        try:
            resp = requests.get(bz_url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            with open(tmp_bz, "wb") as fh:
                for chunk in tqdm(resp.iter_content(chunk_size=8192), total=(total // 8192) + 1, unit="KB"):
                    if chunk:
                        fh.write(chunk)
            print(f"[INFO] Decompressing {tmp_bz} -> {out_path}")
            with open(tmp_bz, "rb") as fh_in:
                compressed = fh_in.read()
            decompressed = bz2.decompress(compressed)
            with open(out_path, "wb") as fh_out:
                fh_out.write(decompressed)
            os.remove(tmp_bz)
            print(f"[OK] Saved model to {out_path}")
        except Exception as e:
            if os.path.exists(tmp_bz):
                os.remove(tmp_bz)
            print("[ERROR] Failed to download/decompress model:", e)
            raise

    download_and_decompress(SHAPE_PRED_URL, os.path.join(MODEL_DIR, SHAPE_PRED_FILENAME))
    download_and_decompress(FACE_REC_URL, os.path.join(MODEL_DIR, FACE_REC_FILENAME))


def connect_mongo():
    """Return (db, fs) connected to MongoDB Atlas."""
    if not MONGO_USER or not MONGO_PASS or not MONGO_HOST:
        raise RuntimeError("Set MONGO_USER, MONGO_PASS and MONGO_HOST in the script or env vars.")
    encoded_pass = quote_plus(MONGO_PASS)
    uri = f"mongodb+srv://{MONGO_USER}:{encoded_pass}@{MONGO_HOST}/?retryWrites=true&w=majority"
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    # try ping
    client.admin.command('ping')
    db = client[MONGO_DBNAME]
    fs = GridFS(db, collection="face_assets")
    return db, fs


# ---- TTS helper (optional) ----
try:
    import pyttsx3
    tts = pyttsx3.init()
    def speak(text):
        try:
            tts.say(text); tts.runAndWait()
        except Exception:
            pass
except Exception:
    def speak(text):
        pass  # TTS not available; silent fallback

# ---- Load models (after ensure_models) ----
def load_dlib_models():
    sp_path = os.path.join(MODEL_DIR, SHAPE_PRED_FILENAME)
    fr_path = os.path.join(MODEL_DIR, FACE_REC_FILENAME)
    if not os.path.exists(sp_path) or not os.path.exists(fr_path):
        raise FileNotFoundError("dlib model files not found. Run ensure_models().")
    shape_predictor = dlib.shape_predictor(sp_path)
    face_rec_model = dlib.face_recognition_model_v1(fr_path)
    detector = dlib.get_frontal_face_detector()
    return detector, shape_predictor, face_rec_model

def estimate_yaw(shape):
    """Rough yaw estimate using eye positions (degrees)."""
    left_eye = np.mean([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)], axis=0)
    right_eye = np.mean([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)], axis=0)
    eye_line = right_eye - left_eye
    yaw = np.degrees(np.arctan2(eye_line[1], eye_line[0]))
    return yaw

def extract_embedding(frame, detector, shape_predictor, face_rec_model):
    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ensure type
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    dets = detector(rgb, 1)
    if not dets:
        return None, None
    # pick largest detection
    rect = max(dets, key=lambda r: r.width() * r.height()) if isinstance(dets[0], dlib.rectangle) else dets[0]
    # ensure rect type
    if hasattr(rect, "rect"):
        rect = rect.rect
    try:
        shape = shape_predictor(rgb, rect)
        yaw = estimate_yaw(shape)
        face_chip = dlib.get_face_chip(rgb, shape, size=150)
        desc = face_rec_model.compute_face_descriptor(face_chip)
        return np.array(desc, dtype=np.float32), yaw
    except Exception:
        return None, None

def gridfs_put_bytes(fs, data_bytes, filename, metadata=None):
    if metadata is None: metadata = {}
    return fs.put(io.BytesIO(data_bytes), filename=filename, metadata=metadata)

def make_pca_and_store(fs, embeddings, labels, enrollment):
    """Make 3D PCA plot and store into GridFS. Returns file id."""
    if embeddings.shape[0] < 2:
        return None
    pca = PCA(n_components=3)
    comps = pca.fit_transform(embeddings)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(comps[:,0], comps[:,1], comps[:,2])
    for i, lab in enumerate(labels):
        ax.text(comps[i,0], comps[i,1], comps[i,2], lab)
    ax.set_title(f"PCA – {enrollment}")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return gridfs_put_bytes(fs, buf.read(), filename=f"{enrollment}_pca.png", metadata={"enrollment": enrollment, "type":"pca"})

def capture_posture(cap, posture, enrollment, detector, sp, fr):
    speak(f"Please position your head: {posture}")
    print(f"\n➡ Capturing {posture} — hold steady (up to {MAX_POSTURE_DURATION_SEC}s)...")
    start = time.time()
    success_count = 0
    captured_emb = None
    captured_img_id = None

    while time.time() - start < MAX_POSTURE_DURATION_SEC:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        cv2.putText(frame, f"{posture.upper()} | Hold still", (10,30), FONT, 0.7, (0,255,0), 2)
        cv2.imshow("Registration - Press ESC to abort", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return None, None, True  # aborted

        emb, yaw = extract_embedding(frame, detector, sp, fr)
        if emb is not None:
            # check yaw relative to posture
            ok = False
            if posture == "front" and abs(yaw) < 12:
                ok = True
            elif posture == "left" and yaw < -5:
                ok = True
            elif posture == "right" and yaw > 5:
                ok = True
            elif posture == "up":  # small heuristic: eyes higher relative etc — accept any detection
                ok = True
            elif posture == "down":
                ok = True

            if ok:
                success_count += 1
            else:
                success_count = 0

            if success_count >= MIN_SUCCESS_FRAMES:
                # store image bytes and return embedding
                ok_enc, buf = cv2.imencode('.jpg', frame)
                if ok_enc:
                    return emb, buf.tobytes(), False
                else:
                    return emb, None, False

    speak(f"Could not capture {posture}.")
    print(f"❌ {posture} not captured.")
    return None, None, False

def register_student():
    try:
        print("=== Student Registration ===")
        name = input("Enter Name: ").strip()
        enrollment = input("Enter Enrollment No: ").strip()
        gmail = input("Enter Gmail (@gmail.com): ").strip()

        # simple validation
        if not enrollment.isalnum():
            print("Enrollment must be alphanumeric.")
            return
        if not gmail.endswith("@gmail.com"):
            print("Gmail must end with @gmail.com")
            return

        # ensure models and load them
        ensure_models()
        detector, shape_predictor, face_rec_model = load_dlib_models()

        # connect mongo
        db, fs = connect_mongo()
        users_col = db["users"]

        # open cam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not found or cannot be opened. Check permissions and camera index.")
            return

        # warm up
        for _ in range(8):
            cap.read()
            time.sleep(0.02)

        embeddings = []
        emb_dict = {}
        img_grid_ids = {}
        failed_postures = []
        aborted = False

        for posture in POSTURES:
            emb, img_bytes, was_aborted = capture_posture(cap, posture, enrollment, detector, shape_predictor, face_rec_model)
            if was_aborted:
                aborted = True
                break
            if emb is None:
                failed_postures.append(posture)
            else:
                embeddings.append(emb)
                emb_dict[posture] = emb.tolist()
                if img_bytes:
                    fid = gridfs_put_bytes(fs, img_bytes, filename=f"{enrollment}_{posture}.jpg",
                                           metadata={"enrollment": enrollment, "posture": posture})
                    img_grid_ids[posture] = fid

        cap.release()
        cv2.destroyAllWindows()

        if aborted:
            print("Registration aborted by user.")
            return

        if not emb_dict:
            print("❌ Registration failed: no valid postures captured.")
            return

        emb_matrix = np.vstack([np.array(emb_dict[p], dtype=np.float32) for p in emb_dict])
        mean_emb = emb_matrix.mean(axis=0)

        # create PCA plot and store in GridFS
        pca_id = make_pca_and_store(fs, emb_matrix, list(emb_dict.keys()), enrollment)

        doc = {
            "name": name,
            "enrollment": enrollment,
            "gmail": gmail,
            "embeddings": emb_dict,
            "mean_embedding": mean_emb.tolist(),
            "image_gridfs_ids": img_grid_ids,
            "pca_gridfs_id": pca_id,
            "failed_postures": failed_postures,
            "created_at": datetime.utcnow()
        }
        users_col.update_one({"enrollment": enrollment}, {"$set": doc}, upsert=True)

        print("\n🎉 Registration Complete.")
        print(f"Student: {name} ({enrollment})")
        if failed_postures:
            print(f"⚠ Failed postures: {failed_postures}")
        else:
            print("✅ All postures registered successfully.")
        print("✅ Embeddings + PCA plot stored in MongoDB Atlas (GridFS).")

    except Exception as e:
        print("[ERROR] Exception during registration:")
        traceback.print_exc()

if __name__ == "__main__":
    register_student()
