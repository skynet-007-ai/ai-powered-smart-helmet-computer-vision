import cv2
import torch
import pyttsx3
import threading
from ultralytics import YOLO

# --- SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# Load the fast OpenVINO model
try:
    # BONUS FIX: Added task='detect' to remove the warning
    model = YOLO("yolov8n_openvino_model/", task='detect')
    print("✅ OpenVINO model loaded successfully for testing.")
except Exception as e:
    print(f"❌ Error loading OpenVINO model: {e}")
    exit()

# TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Alerts per class (lowercase keys)
alerts = {
    "stop sign": "Stop sign ahead. Please slow down.",
    "traffic light": "Traffic light detected ahead.",
    "fire hydrant": "Fire hydrant detected ahead.",
}

# State management
active_alerts = set()
alert_lock = threading.Lock()
frame_counter = 0
# 1. Create a lock for the speech engine
speech_lock = threading.Lock()

# --- ADJUSTED FOR LAPTOP PERFORMANCE ---
PROCESS_EVERY_N_FRAMES = 1
IMAGE_SIZE = (640, 480)


def speak_alert(message):
    def run():
        # 2. Use the lock to ensure only one thread speaks at a time
        with speech_lock:
            print(f"🔊 Alert: {message}")
            engine.say(message)
            engine.runAndWait()

    thread = threading.Thread(target=run)
    thread.start()


# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()

cap.set(3, 640)
cap.set(4, 480)

print("\n🚀 Starting test on Laptop. Press 'q' to quit.")

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    annotated_frame = frame.copy()

    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        resized_frame = cv2.resize(frame, IMAGE_SIZE)
        results = model(resized_frame, device=device, conf=0.45, verbose=False)

        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        detected_classes = {model.names[int(cls)].lower().strip() for cls in class_ids}

        current_alertable_objects = {cls for cls in detected_classes if cls in alerts}

        with alert_lock:
            new_alerts_to_speak = current_alertable_objects - active_alerts

        for cls in new_alerts_to_speak:
            speak_alert(alerts[cls])

        with alert_lock:
            active_alerts = current_alertable_objects

        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

    cv2.imshow("AI Helmet - Laptop Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()