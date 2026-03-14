import cv2
import time
import torch
import pyttsx3
import threading
from ultralytics import YOLO

# ------------------------------
# DEVICE SETUP
# ------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# ------------------------------
# LOAD YOLO MODEL
# ------------------------------

model = YOLO("D:\My Projects\AI Powered Helmet Project\AI Powered Helmet For Road Safety\models\yolov8n.pt")
model.to(device)

# ------------------------------
# TEXT TO SPEECH SETUP
# ------------------------------

engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

speech_lock = threading.Lock()

# ------------------------------
# ALERT MESSAGES
# ------------------------------

alerts = {
    "person": "Pedestrian detected ahead.",
    "bicycle": "Cyclist nearby.",
    "car": "Car approaching.",
    "motorcycle": "Motorbike nearby.",
    "bus": "Bus ahead.",
    "truck": "Truck nearby.",
    "traffic light": "Traffic light ahead.",
    "stop sign": "Stop sign detected. Please slow down.",
    "cow": "Animal on road ahead.",
    "dog": "Dog detected on road."
}

# ------------------------------
# ALERT MANAGEMENT
# ------------------------------

alert_times = {}
cooldown = 4  # seconds before repeating same alert

def speak_alert(message):
    now = time.time()

    if now - alert_times.get(message, 0) < cooldown:
        return

    alert_times[message] = now

    def run():
        with speech_lock:
            print(f"🔊 Alert: {message}")
            engine.say(message)
            engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()


# ------------------------------
# CAMERA SETUP
# ------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

cap.set(3, 640)
cap.set(4, 480)

print("\n🚦 AI Helmet Vision started. Press 'q' to quit.")

# ------------------------------
# FRAME TRACKING FOR STABILITY
# ------------------------------

frame_history = {}
required_frames = 3  # object must appear in 3 frames before alert

# ------------------------------
# MAIN LOOP
# ------------------------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster inference
    resized = cv2.resize(frame, (640, 480))

    # YOLO inference
    results = model(resized, device=device, conf=0.5, verbose=False)

    boxes = results[0].boxes

    detected_classes = []

    if boxes.cls is not None:
        class_ids = boxes.cls.cpu().numpy()
        detected_classes = [model.names[int(cls)].lower().strip() for cls in class_ids]

    current_objects = set(detected_classes)

    # ------------------------------
    # STABILITY CHECK
    # ------------------------------

    for obj in current_objects:

        frame_history[obj] = frame_history.get(obj, 0) + 1

        if frame_history[obj] == required_frames:
            if obj in alerts:
                speak_alert(alerts[obj])

    # Reset objects not detected this frame
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