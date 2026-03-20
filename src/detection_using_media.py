import cv2
import time
import torch
import os
import threading
import queue
from collections import Counter, deque
from ultralytics import YOLO

# ------------------------------
# DEVICE
# ------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

model = YOLO(r"D:\My Projects\AI Powered Helmet Project\AI Powered Helmet For Road Safety\models\yolov8n.pt")
model.to(device)

# ------------------------------
# ALERT GENERATION
# ------------------------------

def generate_alert(obj, count):
    if count == 1:
        return f"{obj.capitalize()} ahead."
    else:
        return f"Multiple {obj}s ahead."

# ------------------------------
# SPEECH QUEUE
# ------------------------------

speech_queue = queue.Queue()

def speech_worker():
    while True:
        msg = speech_queue.get()
        if msg is None:
            break

        print(f"🔊 {msg}")

        command = f'''powershell -Command "Add-Type -AssemblyName System.Speech; \
        $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
        $speak.Speak('{msg}')"'''

        os.system(command)
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

# ------------------------------
# CONTROL
# ------------------------------

COOLDOWN = 4
CONF_THRESHOLD = 0.5

last_spoken = {}
history = deque(maxlen=10)   # last 10 frames memory

# ------------------------------
# VIDEO
# ------------------------------

cap = cv2.VideoCapture(r"D:\My Projects\AI Powered Helmet Project\Video Project.mp4")

print("\n🚦 AI Helmet Vision started. Press 'q' to quit.")

# ------------------------------
# LOOP
# ------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    resized = cv2.resize(frame, (640, 480))
    results = model(resized, device=device, conf=CONF_THRESHOLD, verbose=False)

    boxes = results[0].boxes
    current_time = time.time()

    detected_classes = []

    if boxes.cls is not None:
        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        for cls, conf in zip(class_ids, confidences):
            if conf > CONF_THRESHOLD:
                detected_classes.append(model.names[int(cls)].lower().strip())

    # ------------------------------
    # TEMPORAL SMOOTHING
    # ------------------------------

    history.append(detected_classes)

    # Flatten history
    all_recent = [item for sublist in history for item in sublist]
    counts = Counter(all_recent)

    # Only keep strong signals
    stable_counts = {k: v for k, v in counts.items() if v > 5}

    # ------------------------------
    # ALERT LOGIC
    # ------------------------------

    for obj, count in stable_counts.items():

        alert_msg = generate_alert(obj, count)

        if current_time - last_spoken.get(obj, 0) > COOLDOWN:
            speech_queue.put(alert_msg)
            last_spoken[obj] = current_time

    # ------------------------------
    # DISPLAY
    # ------------------------------

    annotated = results[0].plot()
    cv2.imshow("AI Smart Helmet Vision", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ------------------------------
# CLEANUP
# ------------------------------

cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)