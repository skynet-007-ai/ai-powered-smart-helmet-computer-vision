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
    # "horn prohibited": "Horn prohibited zone.",
    # "school bus": "School bus nearby. Drive carefully.",
    "fire hydrant": "Fire hydrant detected ahead.",
    "parking meter": "Parking meter ahead.",
    # "no entry": "No entry ahead.",
    # "barrier": "Barrier ahead. Please be cautious.",
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

print("\n🚦 AI Helmet Vision started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference on the original frame for better accuracy
    results = model(frame, device=device, verbose=False)
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
    detected_classes = [model.names[int(cls)].lower().strip() for cls in class_ids]

    # Count persons
    person_count = detected_classes.count("person")

    # Use set for unique detected classes except 'person'
    unique_classes = set(detected_classes)
    if "person" in unique_classes:
        unique_classes.remove("person")

    # --- START: REPLACEMENT LOGIC ---

    # 1. Identify all alertable objects in the current frame
    current_alertable_objects = {cls for cls in unique_classes if cls in alerts}

    # 2. Add 'pedestrians' as a special alertable object if condition is met
    if person_count >= 3:
        current_alertable_objects.add("pedestrians")

    # 3. Determine which alerts are genuinely new by comparing with the active set
    with alert_lock:
        new_alerts_to_speak = current_alertable_objects - active_alerts

    # 4. Speak only the new alerts
    for cls in new_alerts_to_speak:
        if cls == "pedestrians":
            # Handle the special pedestrian message
            speak_alert("Multiple pedestrians ahead. Drive carefully.")
        else:
            # Handle standard messages from the dictionary
            speak_alert(alerts[cls])

    # 5. CRITICAL STEP: Update active_alerts to reflect exactly what's visible now.
    # This prepares it for the next frame.
    with alert_lock:
        active_alerts = current_alertable_objects

    # --- END: REPLACEMENT LOGIC ---

    # Display annotated frame
    annotated = results[0].plot()
    cv2.imshow("AI Helmet Vision", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()