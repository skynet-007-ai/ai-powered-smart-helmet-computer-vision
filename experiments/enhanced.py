import cv2
import time
import torch
import pyttsx3
import threading
from ultralytics import YOLO

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# Load YOLO model
model = YOLO("yolov8n.pt")
model.to(device)

# TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Alerts per class (lowercase keys)
alerts = {
    "stop sign": "Stop sign ahead. Please slow down.",
    "traffic light": "Traffic light detected ahead.",
    "horn prohibited": "Horn prohibited zone.",
    "school bus": "School bus nearby. Drive carefully.",
    "fire hydrant": "Fire hydrant detected ahead.",
    "parking meter": "Parking meter ahead.",
    "no entry": "No entry ahead.",
    "barrier": "Barrier ahead. Please be cautious.",
    # Add more road objects if needed
}

# For tracking classes already announced
active_alerts = set()
alert_lock = threading.Lock()

# Speak alert (non-blocking)
def speak_alert(message):
    def run():
        print(f"🔊 Alert: {message}")
        engine.say(message)
        engine.runAndWait()
    thread = threading.Thread(target=run)
    thread.start()

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()

print("\n🚦 Helmet detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    resized_frame = cv2.resize(frame, (320, 240))

    # Inference
    results = model(resized_frame, device=device)
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
    detected_classes = [model.names[int(cls)].lower().strip() for cls in class_ids]

    # Count persons
    person_count = detected_classes.count("person")

    # Use set for unique detected classes except 'person'
    unique_classes = set(detected_classes)
    if "person" in unique_classes:
        unique_classes.remove("person")

    # Determine new alerts
    new_detections = set()

    # Alert for pedestrian only if 3 or more persons
    if person_count >= 3:
        if "pedestrians" not in active_alerts:
            new_detections.add("pedestrians")

    # Check other classes for new alerts
    for cls in unique_classes:
        if cls in alerts and cls not in active_alerts:
            new_detections.add(cls)

    # Speak new alerts
    for cls in new_detections:
        if cls == "pedestrians":
            speak_alert("Multiple pedestrians ahead. Drive carefully.")
        else:
            speak_alert(alerts[cls])

    # Update active alerts (keep only currently detected)
    with alert_lock:
        if person_count < 3 and "pedestrians" in active_alerts:
            active_alerts.remove("pedestrians")

        active_alerts = {cls for cls in active_alerts if (cls in unique_classes or cls == "pedestrians")}
        active_alerts.update(new_detections)

    # Display annotated frame
    annotated = results[0].plot()
    cv2.imshow("YOLO Helmet Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
