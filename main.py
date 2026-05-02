import cv2
import pygame
import numpy as np
import os
import urllib.request
import time

# ════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════
KNOWN_FACES_DIR = "known_faces"
SIREN_FILE = "siren.mp3"
CAMERA_INDEX = 0
PROCESS_EVERY = 2
CONFIRM_FRAMES = 3

SFACE_THRESHOLD = 0.30    # Cosine similarity (lowered from 0.45)
LBPH_THRESHOLD = 55       # Distance (raised from 35)

# ════════════════════════════════════════════════════
# CHECK: opencv-contrib installed properly?
# ════════════════════════════════════════════════════
if not hasattr(cv2, 'face'):
    print("[ERROR] cv2.face not available!")
    print("FIX: python -m pip uninstall opencv-python -y")
    print("     python -m pip install opencv-contrib-python==4.13.0.92")
    exit()

# Check siren file
if not os.path.exists(SIREN_FILE):
    print(f"[ERROR] '{SIREN_FILE}' not found! Place a siren MP3 in the project folder.")
    exit()

# ════════════════════════════════════════════════════
# MODEL DOWNLOADS
# ════════════════════════════════════════════════════
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CAFFE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
CAFFE_PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
SFACE_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")


def download(url, path):
    if os.path.exists(path):
        return True
    print(f"  ⬇️ Downloading {os.path.basename(path)}...")
    try:
        urllib.request.urlretrieve(url, path)
        if os.path.getsize(path) < 10240:
            os.remove(path)
            return False
        print(f"  ✅ {os.path.basename(path)}")
        return True
    except Exception as e:
        print(f"  ❌ {e}")
        if os.path.exists(path):
            os.remove(path)
        return False


# ════════════════════════════════════════════════════
# DETECTION: CAFFE SSD (100% RELIABLE)
# ════════════════════════════════════════════════════
print("[INFO] Loading Caffe SSD detector...")
d1 = download(
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    CAFFE_MODEL,
)
d2 = download(
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    CAFFE_PROTOTXT,
)
if not (d1 and d2):
    print("[ERROR] Detection model download failed!")
    exit()

detect_net = cv2.dnn.readNetFromCaffe(CAFFE_PROTOTXT, CAFFE_MODEL)
print("[INFO] ✅ Caffe SSD detector ready")


def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    detect_net.setInput(blob)
    det = detect_net.forward()
    faces = []
    for i in range(det.shape[2]):
        conf = det[0, 0, i, 2]
        if conf > 0.5:
            box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            fw, fh = x2 - x1, y2 - y1
            if fw > 40 and fh > 40:
                faces.append((x1, y1, fw, fh))
    return faces


# ════════════════════════════════════════════════════
# FACE ALIGNMENT (EYE-BASED) — FIXED
# ════════════════════════════════════════════════════
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def align_face(frame, x, y, w, h):
    """Rotate face so eyes are horizontal. Key for SFace accuracy."""
    pad = int(max(w, h) * 0.3)
    fy1 = max(0, y - pad)
    fy2 = min(frame.shape[0], y + h + pad)
    fx1 = max(0, x - pad)
    fx2 = min(frame.shape[1], x + w + pad)

    region = frame[fy1:fy2, fx1:fx2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15)
    )

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        le_x = float(eyes[0][0] + eyes[0][2] // 2 + fx1)
        le_y = float(eyes[0][1] + eyes[0][3] // 2 + fy1)
        re_x = float(eyes[1][0] + eyes[1][2] // 2 + fx1)
        re_y = float(eyes[1][1] + eyes[1][3] // 2 + fy1)

        dy = re_y - le_y
        dx = re_x - le_x
        angle = np.degrees(np.arctan2(dy, dx))

        if abs(angle) < 30:
            cx = (le_x + re_x) / 2.0
            cy = (le_y + re_y) / 2.0
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            crop = rotated[y:y + h, x:x + w]
            if crop.size > 0:
                return crop

    # Fallback: raw crop
    return frame[y:y + h, x:x + w]


# ════════════════════════════════════════════════════
# RECOGNITION: TRY SFACE VIA cv2.dnn
# ════════════════════════════════════════════════════
USE_SFACE = False
recog_net = None

print("[INFO] Loading SFace recognizer...")
if download(
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    SFACE_MODEL,
):
    try:
        recog_net = cv2.dnn.readNetFromONNX(SFACE_MODEL)
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(
            dummy, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5)
        )
        recog_net.setInput(blob)
        out = recog_net.forward()
        if out.shape[1] == 128:
            USE_SFACE = True
            print("[INFO] ✅ SFace ready (128-d embeddings)")
        else:
            raise Exception(f"Bad shape: {out.shape}")
    except Exception as e:
        print(f"[INFO] ❌ SFace failed: {e}")

# ════════════════════════════════════════════════════
# FALLBACK: LBPH
# ════════════════════════════════════════════════════
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lbph_recognizer = None

if not USE_SFACE:
    print("[INFO] Using LBPH (strict + CLAHE)")
    lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )


# ════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════
def get_embedding(face_img):
    face = cv2.resize(face_img, (112, 112))
    blob = cv2.dnn.blobFromImage(
        face, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5)
    )
    recog_net.setInput(blob)
    return recog_net.forward().flatten()


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def preprocess_lbph(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eq = CLAHE.apply(gray)
    return cv2.resize(eq, (200, 200))


def augment_lbph(face):
    v = [face]
    v.append(cv2.flip(face, 1))
    v.append(cv2.convertScaleAbs(face, alpha=1.2, beta=10))
    v.append(cv2.convertScaleAbs(face, alpha=0.8, beta=-10))
    v.append(cv2.GaussianBlur(face, (3, 3), 0))
    v.append(cv2.convertScaleAbs(face, alpha=1.1, beta=15))
    v.append(cv2.convertScaleAbs(face, alpha=0.9, beta=-15))
    k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    v.append(cv2.filter2D(face, -1, k))
    v.append(cv2.convertScaleAbs(face, alpha=1.15, beta=5))
    v.append(cv2.convertScaleAbs(face, alpha=0.85, beta=-5))
    return v


# ════════════════════════════════════════════════════
# LOAD KNOWN FACES
# ════════════════════════════════════════════════════
print(f"\n[INFO] Loading faces from '{KNOWN_FACES_DIR}/'...")

known_embeddings = []
lbph_faces = []
lbph_labels = []

for filename in sorted(os.listdir(KNOWN_FACES_DIR)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    img = cv2.imread(filepath)
    if img is None:
        print(f"  ⚠️ Cannot read: {filename}")
        continue

    faces = detect_faces(img)
    if not faces:
        print(f"  ⚠️ No face: {filename}")
        continue

    areas = [(fw * fh, (x, y, fw, fh)) for (x, y, fw, fh) in faces]
    _, (x, y, fw, fh) = max(areas)

    aligned = align_face(img, x, y, fw, fh)
    if aligned.size == 0:
        aligned = img[y:y + h, x:x + fw]

    if USE_SFACE:
        try:
            emb = get_embedding(aligned)
            known_embeddings.append(emb)
            # Also store flipped version
            flipped = cv2.flip(aligned, 1)
            emb_f = get_embedding(flipped)
            known_embeddings.append(emb_f)
            print(f"  ✅ {filename} → 2 embeddings")
        except Exception as e:
            print(f"  ⚠️ {filename}: {e}")

    # Always prepare LBPH data too (as backup)
    processed = preprocess_lbph(aligned)
    for aug in augment_lbph(processed):
        lbph_faces.append(aug)
        lbph_labels.append(0)

if USE_SFACE and not known_embeddings:
    print("[WARN] No SFace embeddings. Falling back to LBPH.")
    USE_SFACE = False

if not USE_SFACE:
    if not lbph_faces:
        print("[ERROR] No faces loaded! Run capture_faces.py first.")
        exit()
    lbph_recognizer.train(lbph_faces, np.array(lbph_labels))
    print(f"[INFO] LBPH trained: {len(lbph_faces)} samples, threshold: {LBPH_THRESHOLD}")

mode = "SFace + Eye Alignment" if USE_SFACE else "LBPH + CLAHE"
total_refs = len(known_embeddings) if USE_SFACE else len(lbph_faces)
print(f"\n[INFO] 🧠 Mode: {mode}")
print(f"[INFO] 📚 References: {total_refs}")
print(f"[INFO] 🎯 Threshold: {'%.2f' % SFACE_THRESHOLD if USE_SFACE else LBPH_THRESHOLD}")
print(f"[INFO] Keys: +/- adjust threshold | q quit")
print(f"[INFO] Starting camera...\n")


# ════════════════════════════════════════════════════
# SIREN
# ════════════════════════════════════════════════════
pygame.mixer.init()
siren_sound = pygame.mixer.Sound(SIREN_FILE)
siren_playing = False


def start_siren():
    global siren_playing
    if not siren_playing:
        siren_sound.play(loops=-1)
        siren_playing = True


def stop_siren():
    global siren_playing
    if siren_playing:
        siren_sound.stop()
        siren_playing = False


# ════════════════════════════════════════════════════
# MATCHING
# ════════════════════════════════════════════════════
def match_face(frame, x, y, fw, fh):
    """Returns (is_known, score, label_text)."""
    aligned = align_face(frame, x, y, fw, fh)
    if aligned.size == 0:
        aligned = frame[y:y + fh, x:x + fw]

    if USE_SFACE:
        try:
            emb = get_embedding(aligned)
            best = -1.0
            for ke in known_embeddings:
                s = cosine_sim(emb, ke)
                if s > best:
                    best = s
            is_known = best >= SFACE_THRESHOLD
            return is_known, best, f"cos={best:.3f}"
        except Exception:
            return False, 0.0, "err"
    else:
        processed = preprocess_lbph(aligned)
        label, dist = lbph_recognizer.predict(processed)
        is_known = dist < LBPH_THRESHOLD
        return is_known, dist, f"dist={dist:.1f}"


# ════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Cannot open camera!")
    exit()

time.sleep(1)

frame_count = 0
current_state = "no_face"
confirmed_state = "no_face"
saved_boxes = []
confirm_counter = {"known": 0, "unknown": 0}
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    fps = 0.9 * fps + 0.1 / (now - prev_time + 1e-10)
    prev_time = now

    frame_count += 1
    process_this = frame_count % PROCESS_EVERY == 0

    if process_this:
        faces = detect_faces(frame)

        if not faces:
            current_state = "no_face"
            saved_boxes = []
        else:
            known_found = False
            boxes = []

            for x, y, fw, fh in faces:
                is_known, score, label = match_face(frame, x, y, fw, fh)
                boxes.append((x, y, fw, fh, is_known, label))
                if is_known:
                    known_found = True

                # ══ DEBUG: print to console ══
                tag = "✅ KNOWN" if is_known else "❌ UNKNOWN"
                thr = f"thr={'%.2f' % SFACE_THRESHOLD}" if USE_SFACE else f"thr={LBPH_THRESHOLD}"
                print(f"  {tag} | {label} | {thr}")

            if known_found:
                current_state = "known"
            elif boxes:
                current_state = "unknown"
            else:
                current_state = "no_face"

            saved_boxes = boxes

        # Temporal smoothing
        if current_state == "no_face":
            confirmed_state = "no_face"
            confirm_counter = {"known": 0, "unknown": 0}
        elif current_state == "known":
            confirm_counter["known"] += 1
            confirm_counter["unknown"] = 0
            if confirm_counter["known"] >= CONFIRM_FRAMES:
                confirmed_state = "known"
        elif current_state == "unknown":
            confirm_counter["unknown"] += 1
            confirm_counter["known"] = 0
            if confirm_counter["unknown"] >= CONFIRM_FRAMES:
                confirmed_state = "unknown"

        # Siren
        if confirmed_state in ("known", "no_face"):
            stop_siren()
        elif confirmed_state == "unknown":
            start_siren()

    # ══════════════════════════════════════════════
    # DRAW
    # ══════════════════════════════════════════════
    display = frame.copy()

    for x, y, fw, fh, is_known, label in saved_boxes:
        color = (0, 255, 0) if is_known else (0, 0, 255)
        text = f"{'KNOWN' if is_known else 'UNKNOWN'} | {label}"

        cv2.rectangle(display, (x, y), (x + fw, y + fh), color, 2)
        cv2.rectangle(display, (x, y + fh), (x + fw, y + fh + 30), color, cv2.FILLED)
        cv2.putText(
            display, text, (x + 4, y + fh + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    # Top bar
    if confirmed_state == "known":
        status = "KNOWN — Safe"
        scolor = (0, 255, 0)
    elif confirmed_state == "unknown":
        status = "UNKNOWN — SIREN!"
        scolor = (0, 0, 255)
    else:
        status = "No Face"
        scolor = (200, 200, 200)

    cv2.rectangle(display, (0, 0), (display.shape[1], 72), (0, 0, 0), cv2.FILLED)
    cv2.putText(display, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, scolor, 2)
    cv2.putText(display, f"{mode} | FPS:{fps:.0f} | +/- threshold | q quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Bottom: threshold
    t_text = f"Threshold: {'%.2f' % SFACE_THRESHOLD if USE_SFACE else LBPH_THRESHOLD}"
    cv2.putText(display, t_text, (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("Face Security System", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key in (ord("+"), ord("=")):
        if USE_SFACE:
            SFACE_THRESHOLD = min(SFACE_THRESHOLD + 0.02, 0.90)
        else:
            LBPH_THRESHOLD = max(LBPH_THRESHOLD - 3, 10)
        val = f"{SFACE_THRESHOLD:.2f}" if USE_SFACE else f"{LBPH_THRESHOLD}"
        print(f"  ⬆️ Threshold → {val} (stricter)")
    elif key == ord("-"):
        if USE_SFACE:
            SFACE_THRESHOLD = max(SFACE_THRESHOLD - 0.02, 0.10)
        else:
            LBPH_THRESHOLD = min(LBPH_THRESHOLD + 3, 120)
        val = f"{SFACE_THRESHOLD:.2f}" if USE_SFACE else f"{LBPH_THRESHOLD}"
        print(f"  ⬇️ Threshold → {val} (more lenient)")

# Cleanup
stop_siren()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("[INFO] System stopped.")