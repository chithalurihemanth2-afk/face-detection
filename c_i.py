import cv2
import os
import urllib.request
import numpy as np

# ════════════════════════════════════════════════════
KNOWN_FACES_DIR = "known_faces"
CAMERA_INDEX = 0
MAX_PHOTOS = 5
# ════════════════════════════════════════════════════

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CAFFE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
CAFFE_PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")

for path, url in [
    (CAFFE_MODEL, "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"),
    (CAFFE_PROTOTXT, "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"),
]:
    if not os.path.exists(path):
        print(f"  ⬇️ Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)

net = cv2.dnn.readNetFromCaffe(CAFFE_PROTOTXT, CAFFE_MODEL)


def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    det = net.forward()
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


os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
for f in os.listdir(KNOWN_FACES_DIR):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        os.remove(os.path.join(KNOWN_FACES_DIR, f))

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Cannot open camera!")
    exit()

print("=" * 50)
print("  FACE CAPTURE — 5 Photos")
print("  [SPACE] Capture  [Q] Quit")
print("  TIP: Slight expression change each time")
print("=" * 50)

captured = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)
    display = frame.copy()
    ok = len(faces) > 0

    if ok:
        for (x, y, fw, fh) in faces:
            cv2.rectangle(display, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
        cv2.putText(display, f"Face OK — Press SPACE ({MAX_PHOTOS - captured} left)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No face — look at camera",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(display, f"Captured: {captured}/{MAX_PHOTOS}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Capture Faces", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and ok:
        captured += 1
        fp = os.path.join(KNOWN_FACES_DIR, f"me_{captured}.jpg")
        cv2.imwrite(fp, frame)
        print(f"  📸 Saved: {fp}")
        flash = display.copy()
        flash[:] = (255, 255, 255)
        cv2.imshow("Capture Faces", flash)
        cv2.waitKey(200)
        if captured >= MAX_PHOTOS:
            print(f"\n  ✅ {MAX_PHOTOS} photos captured!")
            break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"  Next: python main.py")