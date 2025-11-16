# """
# recognition_live.py

# Real-time face recognition using dlib embeddings + MongoDB.
# - Loads registered users from `users` collection (expects fields: enrollment, name, mean_embedding, embeddings {posture: [...]})
# - Detects faces on webcam, computes embeddings, matches against DB using mean + posture logic
# - Draws bounding box + label on webcam preview
# - Inserts attendance documents into `attendance` collection:
#   {
#     enrollment, name, recognized_at (UTC), distance, bbox, method ("mean"|"posture"), raw_time
#   }
# - Avoids duplicate rapid inserts using per-user cooldown
# """

# import os
# import time
# import io
# import sys
# import traceback
# from datetime import datetime, timezone

# import cv2
# import dlib
# import numpy as np
# from pymongo import MongoClient

# # --- CONFIG (edit or set via env vars) ---
# MONGO_USER = os.getenv("MONGO_USER") or "muskanrohe2005"
# MONGO_PASS = os.getenv("MONGO_PASS") or "Harshi@9424"
# MONGO_HOST = os.getenv("MONGO_HOST") or "cluster0.uka2xrp.mongodb.net"
# MONGO_DBNAME = os.getenv("MONGO_DBNAME") or "elearning_platform"

# # dlib model paths (must exist in ./models/)
# MODEL_DIR = "models"
# SHAPE_PRED = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
# FACE_REC_MODEL = os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")

# # Matching thresholds (tune these)
# MEAN_THRESHOLD = 0.60         # euclidean distance to mean embedding
# POSTURE_THRESHOLD = 0.60      # distance to individual posture embedding
# MIN_MATCHED_POSTURES = 2      # number of posture embeddings that must match to accept
# COOLDOWN_SECONDS = 30         # don't log same user again within this many seconds

# # Camera / visualization
# CAMERA_INDEX = 0
# WINDOW_NAME = "Recognition (press ESC to quit)"
# FONT = cv2.FONT_HERSHEY_SIMPLEX

# # ------------------------------------------------
# def connect_mongo():
#     from urllib.parse import quote_plus
#     encoded = quote_plus(MONGO_PASS) if MONGO_PASS else ""
#     uri = f"mongodb+srv://{MONGO_USER}:{encoded}@{MONGO_HOST}/?retryWrites=true&w=majority"
#     client = MongoClient(uri, serverSelectionTimeoutMS=10000)
#     client.admin.command('ping')
#     db = client[MONGO_DBNAME]
#     return db

# def load_dlib_models():
#     if not os.path.exists(SHAPE_PRED) or not os.path.exists(FACE_REC_MODEL):
#         raise FileNotFoundError("dlib model files missing. Place them in 'models/' or run the registration script's downloader.")
#     shape_predictor = dlib.shape_predictor(SHAPE_PRED)
#     face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL)
#     detector = dlib.get_frontal_face_detector()
#     return detector, shape_predictor, face_rec_model

# def compute_embedding_from_rect(rgb_img, rect, shape_predictor, face_rec_model):
#     """Given RGB image and dlib rect, return 128D embedding (np.array)"""
#     shape = shape_predictor(rgb_img, rect)
#     face_chip = dlib.get_face_chip(rgb_img, shape, size=150)
#     desc = face_rec_model.compute_face_descriptor(face_chip)
#     return np.array(desc, dtype=np.float32)

# def euclidean(a, b):
#     return np.linalg.norm(a - b)

# def prepare_users(users_docs):
#     """
#     Build in-memory user structures:
#       users = [
#         {
#           "enrollment": ..., "name": ..., "mean": np.array(...), "postures": {p: [np.array,...]}
#         }, ...
#       ]
#     """
#     users = []
#     for d in users_docs:
#         try:
#             mean = np.array(d.get("mean_embedding") or d.get("mean_emb") or d.get("mean"), dtype=np.float32)
#         except Exception:
#             mean = None
#         postures_raw = d.get("embeddings", {}) or {}
#         postures = {}
#         for p, vec in postures_raw.items():
#             # each vec might be a list (128 floats) or multiple; we assume single vector per posture saved as list
#             try:
#                 postures[p] = [np.array(vec, dtype=np.float32)]
#             except Exception:
#                 # support case where embeddings[p] is list of list
#                 try:
#                     postures[p] = [np.array(v, dtype=np.float32) for v in vec]
#                 except Exception:
#                     postures[p] = []
#         users.append({
#             "enrollment": d.get("enrollment") or d.get("student_id"),
#             "name": d.get("name"),
#             "mean": mean,
#             "postures": postures,
#             "_id": d.get("_id")
#         })
#     return users

# def match_embedding(emb, user):
#     """
#     Returns (method, best_distance, posture_matches_count)
#     method: "mean" or "posture" or None
#     """
#     # mean check
#     if user["mean"] is not None:
#         dist_mean = euclidean(emb, user["mean"])
#         if dist_mean <= MEAN_THRESHOLD:
#             return "mean", float(dist_mean), 0

#     # posture check: count how many stored posture embeddings are within POSTURE_THRESHOLD
#     matched = 0
#     min_dist = float("inf")
#     for p, vecs in user["postures"].items():
#         for v in vecs:
#             d = euclidean(emb, v)
#             if d < min_dist: min_dist = d
#             if d <= POSTURE_THRESHOLD:
#                 matched += 1
#                 break  # count at most once per posture
#     if matched >= MIN_MATCHED_POSTURES:
#         return "posture", float(min_dist), matched

#     return None, float(min_dist if min_dist != float("inf") else 999.0), matched

# def draw_label(frame, rect, text, color=(0,255,0)):
#     left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
#     cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#     cv2.putText(frame, text, (left, top - 10), FONT, 0.6, color, 2)

# def main_loop():
#     db = connect_mongo()
#     users_col = db["users"]
#     attendance_col = db["attendance"]

#     print("[INFO] Loading users from DB...")
#     docs = list(users_col.find({}))
#     users = prepare_users(docs)
#     print(f"[INFO] Loaded {len(users)} users.")

#     detector, shape_predictor, face_rec_model = load_dlib_models()
#     cap = cv2.VideoCapture(CAMERA_INDEX)
#     if not cap.isOpened():
#         print("ERROR: Could not open camera index", CAMERA_INDEX)
#         return

#     # last seen timestamps to prevent repeated logging
#     last_seen = {}  # enrollment -> timestamp (seconds)

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 time.sleep(0.01)
#                 continue

#             # small resize for speed (optional)
#             height, width = frame.shape[:2]
#             scale = 1.0
#             # uncomment to speed up: scale = 0.6
#             if scale != 1.0:
#                 small = cv2.resize(frame, (int(width*scale), int(height*scale)))
#             else:
#                 small = frame

#             rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
#             dets = detector(rgb, 0)

#             # for each detected face, try to compute embedding and match
#             for d in dets:
#                 # if we resized, map rect back
#                 if scale != 1.0:
#                     rect = dlib.rectangle(
#                         int(d.left() / scale),
#                         int(d.top() / scale),
#                         int(d.right() / scale),
#                         int(d.bottom() / scale)
#                     )
#                     rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     try:
#                         emb = compute_embedding_from_rect(rgb_full, rect, shape_predictor, face_rec_model)
#                     except Exception:
#                         continue
#                     rect_to_draw = rect
#                 else:
#                     rect = d
#                     try:
#                         emb = compute_embedding_from_rect(rgb, rect, shape_predictor, face_rec_model)
#                     except Exception:
#                         continue
#                     rect_to_draw = rect

#                 # match against all users; find best user (min dist)
#                 best_user = None
#                 best_method = None
#                 best_dist = 999.0
#                 best_matched_count = 0

#                 for u in users:
#                     method, dist, matched_count = match_embedding(emb, u)
#                     if dist < best_dist:
#                         best_dist = dist
#                         best_user = u
#                         best_method = method
#                         best_matched_count = matched_count

#                 label = "Unknown"
#                 color = (0,0,255)
#                 if best_method is not None and best_user is not None:
#                     label = f"{best_user['name']} ({best_user['enrollment']}) {best_method} d={best_dist:.2f}"
#                     color = (0,255,0)

#                     # attendance logging with cooldown
#                     now_ts = time.time()
#                     enroll = best_user["enrollment"]
#                     last = last_seen.get(enroll, 0)
#                     if now_ts - last >= COOLDOWN_SECONDS:
#                         last_seen[enroll] = now_ts
#                         # create attendance entry
#                         left, top, right, bottom = rect_to_draw.left(), rect_to_draw.top(), rect_to_draw.right(), rect_to_draw.bottom()
#                         attendance_doc = {
#                             "enrollment": enroll,
#                             "name": best_user["name"],
#                             "recognized_at": datetime.now(timezone.utc),
#                             "method": best_method,
#                             "distance": float(best_dist),
#                             "matched_posture_count": int(best_matched_count),
#                             "bbox": {"left": int(left), "top": int(top), "right": int(right), "bottom": int(bottom)},
#                             "raw_time": time.time()
#                         }
#                         try:
#                             attendance_col.insert_one(attendance_doc)
#                             print(f"[LOG] Inserted attendance for {enroll} at {attendance_doc['recognized_at']} (dist {best_dist:.3f})")
#                         except Exception as e:
#                             print("[ERROR] Failed insert attendance:", e)

#                 draw_label(frame, rect_to_draw, label, color=color)

#             cv2.imshow(WINDOW_NAME, frame)
#             key = cv2.waitKey(1) & 0xFF
#             if key == 27:
#                 break

#     except KeyboardInterrupt:
#         print("Interrupted by user.")
#     except Exception as e:
#         print("Exception in main loop:")
#         traceback.print_exc()
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main_loop()













#-----------nice for unknown and multi user also







# """
# recognition_live.py  —  Updated version

# Real-time face recognition using dlib embeddings + MongoDB.
# - Loads registered users from `users` collection
# - Detects faces via webcam, matches against DB using stricter mean+posture rules
# - Labels recognized faces, marks "Unknown" for uncertain or unregistered
# - Logs attendance with cooldown
# """

# import os
# import time
# import traceback
# from datetime import datetime, timezone
# import cv2
# import dlib
# import numpy as np
# from pymongo import MongoClient
# from urllib.parse import quote_plus

# # ===================== CONFIG =====================
# MONGO_USER = os.getenv("MONGO_USER") or "muskanrohe2005"
# MONGO_PASS = os.getenv("MONGO_PASS") or "Harshi@9424"
# MONGO_HOST = os.getenv("MONGO_HOST") or "cluster0.uka2xrp.mongodb.net"
# MONGO_DBNAME = os.getenv("MONGO_DBNAME") or "elearning_platform"

# MODEL_DIR = "models"
# SHAPE_PRED = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
# FACE_REC_MODEL = os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")

# # Tighter thresholds for better accuracy
# MEAN_THRESHOLD = 0.48
# POSTURE_THRESHOLD = 0.48
# MIN_MATCHED_POSTURES = 1
# COOLDOWN_SECONDS = 30

# CAMERA_INDEX = 0
# WINDOW_NAME = "Recognition (press ESC to quit)"
# FONT = cv2.FONT_HERSHEY_SIMPLEX
# # ==================================================


# def connect_mongo():
#     encoded = quote_plus(MONGO_PASS) if MONGO_PASS else ""
#     uri = f"mongodb+srv://{MONGO_USER}:{encoded}@{MONGO_HOST}/?retryWrites=true&w=majority"
#     client = MongoClient(uri, serverSelectionTimeoutMS=10000)
#     client.admin.command("ping")
#     db = client[MONGO_DBNAME]
#     print("[INFO] ✅ Connected to MongoDB successfully.")
#     return db


# def load_dlib_models():
#     if not os.path.exists(SHAPE_PRED) or not os.path.exists(FACE_REC_MODEL):
#         raise FileNotFoundError("dlib model files missing in 'models/'.")
#     shape_predictor = dlib.shape_predictor(SHAPE_PRED)
#     face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL)
#     detector = dlib.get_frontal_face_detector()
#     return detector, shape_predictor, face_rec_model


# def compute_embedding_from_rect(rgb_img, rect, shape_predictor, face_rec_model):
#     shape = shape_predictor(rgb_img, rect)
#     face_chip = dlib.get_face_chip(rgb_img, shape, size=150)
#     desc = face_rec_model.compute_face_descriptor(face_chip)
#     return np.array(desc, dtype=np.float32)


# def euclidean(a, b):
#     return np.linalg.norm(a - b)


# def prepare_users(users_docs):
#     """Load embeddings and mean vectors from MongoDB into memory"""
#     users = []
#     for d in users_docs:
#         try:
#             mean = np.array(d.get("mean_embedding"), dtype=np.float32)
#         except Exception:
#             mean = None
#         postures_raw = d.get("embeddings", {}) or {}
#         postures = {}
#         for p, vec in postures_raw.items():
#             try:
#                 postures[p] = [np.array(vec, dtype=np.float32)]
#             except Exception:
#                 try:
#                     postures[p] = [np.array(v, dtype=np.float32) for v in vec]
#                 except Exception:
#                     postures[p] = []
#         users.append({
#             "enrollment": d.get("enrollment"),
#             "name": d.get("name"),
#             "mean": mean,
#             "postures": postures
#         })
#     return users


# # ---------- Enhanced matching logic ----------
# def match_embedding(emb, user):
#     """Returns (method, best_distance, posture_matches_count)"""
#     dist_mean = 999.0
#     if user["mean"] is not None:
#         dist_mean = euclidean(emb, user["mean"])

#     matched = 0
#     min_dist = float("inf")
#     for _, vecs in user["postures"].items():
#         for v in vecs:
#             d = euclidean(emb, v)
#             if d < min_dist:
#                 min_dist = d
#             if d <= POSTURE_THRESHOLD:
#                 matched += 1
#                 break

#     # Require BOTH mean AND at least one posture match
#     if dist_mean <= MEAN_THRESHOLD and matched >= MIN_MATCHED_POSTURES:
#         return "mean+posture", float(dist_mean), matched

#     return None, float(min_dist if min_dist != float("inf") else 999.0), matched
# # -----------------------------------------------


# def draw_label(frame, rect, text, color=(0, 255, 0)):
#     left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
#     cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#     cv2.putText(frame, text, (left, top - 10), FONT, 0.6, color, 2)


# def main_loop():
#     db = connect_mongo()
#     users_col = db["users"]
#     attendance_col = db["attendance"]

#     print("[INFO] Loading registered users...")
#     users = prepare_users(list(users_col.find({})))
#     print(f"[INFO] Loaded {len(users)} users into memory.")

#     detector, shape_predictor, face_rec_model = load_dlib_models()

#     cap = cv2.VideoCapture(CAMERA_INDEX)
#     if not cap.isOpened():
#         print("❌ Camera not found.")
#         return

#     last_seen = {}  # cooldown tracker

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 time.sleep(0.01)
#                 continue

#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             dets = detector(rgb, 0)

#             for rect in dets:
#                 try:
#                     emb = compute_embedding_from_rect(rgb, rect, shape_predictor, face_rec_model)
#                 except Exception:
#                     continue

#                 best_user, best_method, best_dist, best_matched = None, None, 999.0, 0

#                 for u in users:
#                     method, dist, matched = match_embedding(emb, u)
#                     if dist < best_dist:
#                         best_dist, best_method, best_user, best_matched = dist, method, u, matched

#                 label, color = "Unknown", (0, 0, 255)
#                 if best_user and best_method:
#                     # Uncertain zone
#                     if best_dist > 0.48 and best_dist < 0.55:
#                         label = f"Unknown (uncertain) d={best_dist:.2f}"
#                         color = (0, 255, 255)
#                     else:
#                         label = f"{best_user['name']} ({best_user['enrollment']}) d={best_dist:.2f}"
#                         color = (0, 255, 0)

#                         now_ts = time.time()
#                         enroll = best_user["enrollment"]
#                         last = last_seen.get(enroll, 0)
#                         if now_ts - last >= COOLDOWN_SECONDS:
#                             last_seen[enroll] = now_ts
#                             doc = {
#                                 "enrollment": enroll,
#                                 "name": best_user["name"],
#                                 "recognized_at": datetime.now(timezone.utc),
#                                 "method": best_method,
#                                 "distance": best_dist,
#                                 "matched_posture_count": best_matched,
#                                 "bbox": {
#                                     "left": int(rect.left()), "top": int(rect.top()),
#                                     "right": int(rect.right()), "bottom": int(rect.bottom())
#                                 },
#                                 "raw_time": now_ts
#                             }
#                             try:
#                                 attendance_col.insert_one(doc)
#                                 print(f"[LOG] Attendance recorded for {enroll} at {doc['recognized_at']} (dist {best_dist:.3f})")
#                             except Exception as e:
#                                 print("[ERROR] DB insert failed:", e)

#                 draw_label(frame, rect, label, color=color)

#             cv2.imshow(WINDOW_NAME, frame)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break

#     except KeyboardInterrupt:
#         print("Stopped by user.")
#     except Exception:
#         traceback.print_exc()
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main_loop()

















#--------------------










"""
real_time_attendance.py
Real-time face recognition attendance with entry/exit tracking.
Shows only user name and current time on video,
records entry when face appears and exit when it disappears.
"""

import os
import time
import traceback
from datetime import datetime, timezone
import cv2
import dlib
import numpy as np
from pymongo import MongoClient
from urllib.parse import quote_plus

# ---------------- CONFIG ----------------
MONGO_USER = os.getenv("MONGO_USER") or "muskanrohe2005"
MONGO_PASS = os.getenv("MONGO_PASS") or "Harshi@9424"
MONGO_HOST = os.getenv("MONGO_HOST") or "cluster0.uka2xrp.mongodb.net"
MONGO_DBNAME = os.getenv("MONGO_DBNAME") or "elearning_platform"

MODEL_DIR = "models"
SHAPE_PRED = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
FACE_REC_MODEL = os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")

MEAN_THRESHOLD = 0.48
POSTURE_THRESHOLD = 0.48
MIN_MATCHED_POSTURES = 1
ABSENCE_TIMEOUT = 5  # seconds after which user considered gone

CAMERA_INDEX = 0
WINDOW_NAME = "Real-Time Attendance"
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ----------------------------------------


def connect_mongo():
    encoded = quote_plus(MONGO_PASS) if MONGO_PASS else ""
    uri = f"mongodb+srv://{MONGO_USER}:{encoded}@{MONGO_HOST}/?retryWrites=true&w=majority"
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    print("[INFO] ✅ Connected to MongoDB")
    return client[MONGO_DBNAME]


def load_dlib_models():
    if not os.path.exists(SHAPE_PRED) or not os.path.exists(FACE_REC_MODEL):
        raise FileNotFoundError("Model files missing in 'models/'.")
    shape_predictor = dlib.shape_predictor(SHAPE_PRED)
    face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL)
    detector = dlib.get_frontal_face_detector()
    return detector, shape_predictor, face_rec_model


def compute_embedding_from_rect(rgb_img, rect, shape_predictor, face_rec_model):
    shape = shape_predictor(rgb_img, rect)
    face_chip = dlib.get_face_chip(rgb_img, shape, size=150)
    desc = face_rec_model.compute_face_descriptor(face_chip)
    return np.array(desc, dtype=np.float32)


def euclidean(a, b):
    return np.linalg.norm(a - b)


# def prepare_users(users_docs):
#     users = []
#     for d in users_docs:
#         try:
#             mean = np.array(d.get("mean_embedding"), dtype=np.float32)
#         except Exception:
#             mean = None
#         postures_raw = d.get("embeddings", {}) or {}
#         postures = {}
#         for p, vec in postures_raw.items():
#             try:
#                 postures[p] = [np.array(vec, dtype=np.float32)]
#             except Exception:
#                 try:
#                     postures[p] = [np.array(v, dtype=np.float32) for v in vec]
#                 except Exception:
#                     postures[p] = []
#         users.append({
#             "enrollment": d.get("enrollment"),
#             "name": d.get("name"),
#             "mean": mean,
#             "postures": postures
#         })
#     return users



def prepare_users(users_docs):
    users = []
    for d in users_docs:
        try:
            mean = np.array(d.get("mean_embedding"), dtype=np.float32)
        except Exception:
            mean = None

        postures_raw = d.get("embeddings", {}) or {}
        postures = {}
        for p, vec in postures_raw.items():
            try:
                postures[p] = [np.array(vec, dtype=np.float32)]
            except Exception:
                try:
                    postures[p] = [np.array(v, dtype=np.float32) for v in vec]
                except Exception:
                    postures[p] = []

        users.append({
            "enrollment": str(d.get("enrollment", "")),
            "name": d.get("name"),
            "mean": mean,
            "postures": postures
        })
    return users







def match_embedding(emb, user):
    dist_mean = 999.0
    if user["mean"] is not None:
        dist_mean = euclidean(emb, user["mean"])

    matched = 0
    min_dist = float("inf")
    for _, vecs in user["postures"].items():
        for v in vecs:
            d = euclidean(emb, v)
            if d < min_dist:
                min_dist = d
            if d <= POSTURE_THRESHOLD:
                matched += 1
                break

    if dist_mean <= MEAN_THRESHOLD and matched >= MIN_MATCHED_POSTURES:
        return True, float(dist_mean)
    return False, float(min_dist if min_dist != float("inf") else 999.0)


# def draw_label(frame, rect, name):
#     left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
#     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#     now = datetime.now().strftime("%H:%M:%S")
#     text = f"{name} — {now}"
#     cv2.putText(frame, text, (left, top - 10), FONT, 0.6, (0, 255, 0), 2)
def draw_label(frame, rect, name, enroll):
    left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    now = datetime.now().strftime("%H:%M:%S")
    text = f"{name} | {enroll} | {now}"
    cv2.putText(frame, text, (left, top - 10), FONT, 0.6, (0, 255, 0), 2)


def main_loop():
    db = connect_mongo()
    users_col = db["users"]
    attendance_col = db["attendance"]

    print("[INFO] Loading users from DB...")
    users = prepare_users(list(users_col.find({})))
    print(f"[INFO] Loaded {len(users)} registered users.")

    detector, shape_predictor, face_rec_model = load_dlib_models()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    active_users = {}  # enrollment -> last_seen_time
    session_ids = {}   # enrollment -> attendance doc _id

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector(rgb, 0)
            seen_now = set()

            for rect in dets:
                try:
                    emb = compute_embedding_from_rect(rgb, rect, shape_predictor, face_rec_model)
                except Exception:
                    continue

                best_user, best_dist = None, 999.0
                for u in users:
                    ok, dist = match_embedding(emb, u)
                    if ok and dist < best_dist:
                        best_user, best_dist = u, dist

                if best_user:
                    name = best_user["name"]
                    enroll = best_user["enrollment"]
                    seen_now.add(enroll)
                    draw_label(frame, rect, name, enroll)



                    # If new detection (entry)
                    if enroll not in active_users:
                        active_users[enroll] = time.time()
                        doc = {
                            "enrollment": enroll,
                            "name": name,
                            "entry_time": datetime.now(timezone.utc),
                            "exit_time": None,
                            "active": True
                        }
                        res = attendance_col.insert_one(doc)
                        session_ids[enroll] = res.inserted_id
                        print(f"[ENTRY] {name} entered at {doc['entry_time']}")

                    else:
                        active_users[enroll] = time.time()

            # Handle exits
            now_t = time.time()
            for enroll in list(active_users.keys()):
                if enroll not in seen_now and now_t - active_users[enroll] > ABSENCE_TIMEOUT:
                    # mark exit
                    doc_id = session_ids.get(enroll)
                    if doc_id:
                        attendance_col.update_one(
                            {"_id": doc_id},
                            {"$set": {"exit_time": datetime.now(timezone.utc), "active": False}}
                        )
                    print(f"[EXIT] {enroll} left at {datetime.now(timezone.utc)}")
                    del active_users[enroll]
                    del session_ids[enroll]

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception:
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
