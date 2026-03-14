import cv2
import time
import torch
import pyttsx3
import threading
from ultralytics import YOLO

# --- DEDICATED CAMERA THREAD CLASS ---
# This class runs the camera in a separate thread to prevent stale frames
# and ensure the main loop always gets the most recent picture.
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print("❌ Cannot access webcam.")
            raise IOError("Cannot open webcam")
        
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

model = YOLO("yolov8n.pt", task='detect')
model.to(device)

engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

alerts = {
    'person': 'Watch out, pedestrian.',
    'bicycle': 'Cyclist ahead.',
    'car': 'Careful, car nearby.',
    'motorcycle': 'Motorbike approaching.',
    'bus': 'Heads up, bus ahead.',
    'traffic light': 'Got a traffic light coming up.',
    'stop sign': 'Easy now, stop sign ahead.',
    'cow': 'Whoa! Cow on the road ahead!',
    'dog': 'Careful, dog ahead!',
}

# State management and locks
active_alerts = set()
alert_lock = threading.Lock()
speech_lock = threading.Lock() # Prevents speech engine crashes

# Speak alert with lock
def speak_alert(message):
    def run():
        with speech_lock:
            print(f"🔊 Alert: {message}")
            engine.say(message)
            engine.runAndWait()
    thread = threading.Thread(target=run)
    thread.start()

# --- INITIALIZE AND START THE DEDICATED CAMERA STREAM ---
stream = WebcamStream(src=0).start()
print("\n🚦 AI Helmet Vision started. Press 'q' to quit.")

while True:
    # Grab the LATEST frame from the camera thread
    frame = stream.read()
    if frame is None:
        break

    results = model(frame, device=device, verbose=False)
    
    detected_classes = []
    if results[0].boxes.cls is not None:
        class_ids = results[0].boxes.cls.cpu().numpy()
        detected_classes = [model.names[int(cls)].lower().strip() for cls in class_ids]

    current_alertable_objects = {cls for cls in detected_classes if cls in alerts}

    person_count = detected_classes.count("person")
    if person_count >= 3:
        current_alertable_objects.add("pedestrians")
        if 'person' in current_alertable_objects:
            current_alertable_objects.remove('person')

    with alert_lock:
        new_alerts_to_speak = current_alertable_objects - active_alerts

    for cls in new_alerts_to_speak:
        if cls == "pedestrians":
            speak_alert("Whoa, crowd of people ahead!")
        else:
            speak_alert(alerts[cls])

    with alert_lock:
        active_alerts = current_alertable_objects

    annotated = results[0].plot()
    cv2.imshow("AI Helmet Vision", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
stream.stop()
cv2.destroyAllWindows()