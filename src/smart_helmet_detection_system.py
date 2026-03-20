import cv2
import time
import torch
import os
from ultralytics import YOLO

# ------------------------------
# DEVICE SETUP
# ------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

model = YOLO(r"D:\My Projects\AI Powered Helmet Project\AI Powered Helmet For Road Safety\models\yolov8n.pt")
model.to(device)

# ------------------------------
# ALERTS
# ------------------------------

alerts = {
    "person": "Pedestrian detected ahead, pay attention",
    "bicycle": "Cyclist nearby.",
    "car": "Multiple Cars approaching.",
    "motorcycle": "Motorbike nearby.",
    "bus": "Bus ahead.",
    "truck": "Truck nearby.",
    "traffic light": "Traffic light ahead, follow the signal",
    "stop sign": "Stop sign detected. Please slow down.",
}

# ------------------------------
# ALERT CONTROL
# ------------------------------

last_spoken = {}
cooldown = 3  # seconds

import threading

def speak_alert(message):
    now = time.time()

    if now - last_spoken.get(message, 0) < cooldown:
        return

    last_spoken[message] = now

    print(f"🔊 Alert: {message}")

    def run():
        command = f'''powershell -Command "Add-Type -AssemblyName System.Speech; \
        $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
        $speak.Speak('{message}')"'''

        os.system(command)

    # 🔥 RUN IN BACKGROUND THREAD (NO LAG)
    threading.Thread(target=run, daemon=True).start()

# ------------------------------
# VIDEO INPUT
# ------------------------------

cap = cv2.VideoCapture(r"D:\My Projects\AI Powered Helmet Project\Video Project.mp4")

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

print("\n🚦 AI Helmet Vision started. Press 'q' to quit.")
print("🚀 Running AI Smart Helmet System...")

# ------------------------------
# FRAME TRACKING
# ------------------------------

frame_history = {}
required_frames = 3

# ------------------------------
# MAIN LOOP
# ------------------------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (640, 480))

    results = model(resized, device=device, conf=0.5, verbose=False)

    boxes = results[0].boxes
    detected_classes = []

    if boxes.cls is not None:
        class_ids = boxes.cls.cpu().numpy()
        detected_classes = [model.names[int(cls)].lower().strip() for cls in class_ids]

    current_objects = set(detected_classes)

    # ------------------------------
    # STABILITY LOGIC
    # ------------------------------

    for obj in current_objects:
        frame_history[obj] = frame_history.get(obj, 0) + 1

        if frame_history[obj] == required_frames:
            if obj in alerts:
                speak_alert(alerts[obj])

    # Reset disappeared objects
    for obj in list(frame_history.keys()):
        if obj not in current_objects:
            frame_history[obj] = 0

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