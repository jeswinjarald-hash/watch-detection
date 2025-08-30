import cv2
import time
import threading
import queue
import os
import json
from roboflow import Roboflow

# ====== CONFIG ======
ROBOFLOW_API_KEY = "4lUJ6kk9agNXKo2jnRjY"
WORKSPACE_NAME = "project-b989i"
PROJECT_SLUG = "analog-watches-jmga0"
VERSION_NUMBER = 2
# Accepts either a fraction (0.8) or percentage (80). The code normalizes for the API.
# Lowered for testing — set back to 0.8 when stable.
CONFIDENCE_THRESHOLD = 0.5
OVERLAP_THRESHOLD = 20
VERBOSE = True

# Debug output folder
DEBUG_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'debug_frames')
os.makedirs(DEBUG_DIR, exist_ok=True)

# ====== ROBOFLOW SETUP ======
print("🔐 Connecting to Roboflow...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_SLUG)
model = project.version(VERSION_NUMBER).model

# Quick connectivity / smoke test to ensure the model responds
try:
    black = (255 * (0 * 1)).astype if False else None
    # create a small dummy image (RGB)
    dummy = (255 * 0).astype if False else None
    try:
        import numpy as _np
        dummy = _np.zeros((64, 64, 3), dtype=_np.uint8)
        api_conf = CONFIDENCE_THRESHOLD * 100 if CONFIDENCE_THRESHOLD <= 1 else CONFIDENCE_THRESHOLD
        resp = model.predict(dummy, confidence=api_conf, overlap=OVERLAP_THRESHOLD).json()
        if VERBOSE:
            print(f"🔍 Connectivity check OK — response keys: {list(resp.keys())}")
    except Exception as e:
        print(f"⚠️ Connectivity check failed: {e}")
except Exception:
    pass

# ====== PIPELINE SETUP ======
# Keep only the latest frame for detection to avoid backlog and keep UI smooth.
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

def detection_worker():
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            # Keep an original BGR copy for drawing later
            original_bgr = frame.copy()

            # Convert BGR -> RGB because many models expect RGB input
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Roboflow predict expects confidence as percentage (e.g., 80) in many SDKs.
            api_conf = CONFIDENCE_THRESHOLD * 100 if CONFIDENCE_THRESHOLD <= 1 else CONFIDENCE_THRESHOLD
            result = model.predict(rgb, confidence=api_conf, overlap=OVERLAP_THRESHOLD).json()
            if VERBOSE:
                preds = result.get('predictions', []) if isinstance(result, dict) else []
                print(f"[detector] api_conf={api_conf} raw_preds={len(preds)}")
                if len(preds) > 0:
                    try:
                        print("[detector] first_pred:", json.dumps(preds[0]))
                    except Exception:
                        pass

            # If no predictions, save the frame for offline debugging
            try:
                preds = result.get('predictions', []) if isinstance(result, dict) else []
                if len(preds) == 0 and VERBOSE:
                    ts = int(time.time() * 1000)
                    fname = os.path.join(DEBUG_DIR, f"no_pred_{ts}.jpg")
                    cv2.imwrite(fname, original_bgr)
                    if VERBOSE:
                        print(f"[detector] saved no-detection frame to {fname}")
            except Exception:
                pass

            result_queue.put((original_bgr, result))
        except Exception as e:
            print(f"⚠️ Error in detection: {e}")
            # push the original frame back so display keeps updating
            try:
                result_queue.put((frame, None), timeout=0.1)
            except Exception:
                pass

# ====== START WEBCAM ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("📸 Starting Webcam Detection... Press 'q' to quit.")

frame_count = 0
start_time = time.time()

# Start detection thread
thread = threading.Thread(target=detection_worker, daemon=True)
thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Put frame in queue for detection
    if not frame_queue.full():
        frame_queue.put(frame.copy())

    # Get detection result if available
    if not result_queue.empty():
        frame, result = result_queue.get()
        if result is not None:
            predictions = result['predictions']
            for pred in predictions:
                x, y = int(pred['x']), int(pred['y'])
                w, h = int(pred['width']), int(pred['height'])
                label = pred['class']
                conf = pred['confidence']
                x1 = x - w // 2
                y1 = y - h // 2
                x2 = x + w // 2
                y2 = y + h // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Roboflow Detection", frame)
    else:
        # Show latest frame if no detection result yet
        cv2.imshow("Roboflow Detection", frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== CLEANUP ======
stop_event.set()
thread.join(timeout=1)
end_time = time.time()
fps = frame_count / (end_time - start_time)
print(f"✅ Processed {frame_count} frames in {end_time - start_time:.2f} sec ({fps:.2f} FPS)")

cap.release()
cv2.destroyAllWindows()