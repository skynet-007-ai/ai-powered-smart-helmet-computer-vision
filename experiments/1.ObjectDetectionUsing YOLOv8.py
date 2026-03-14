import cv2
import asyncio
import edge_tts
import time
import os
import pygame
from ultralytics import YOLO

# Initialize model
model = YOLO("yolov8n.pt")

# Alert system config
cooldown_time = 2
alert_times = {}
pygame.mixer.init()

# Async TTS
async def generate_alert(message, voice="en-US-JennyNeural"):
    try:
        communicate = edge_tts.Communicate(text=message, voice=voice)
        await communicate.save("alert.mp3")
    except Exception as e:
        print("TTS generation error:", e)

# Alert playback with cooldown
def speak_alert(message, voice="en-US-JennyNeural"):
    now = time.time()
    if now - alert_times.get(message, 0) > cooldown_time:
        asyncio.run(generate_alert(message, voice))
        try:
            pygame.mixer.music.load("alert.mp3")
            pygame.mixer.music.play()
        except Exception as e:
            print("Audio playback error:", e)
        alert_times[message] = now

# Static class-to-alert mapping
alerts = {
    "Cattle": "Cattle on road ahead",
    "Hospital": "Hospital zone, drive quietly",
    "Horn Prohibited": "Horn prohibited zone",
    "School Ahead": "School zone ahead, drive carefully",
    "Stop": "Stop sign ahead",
    "No Parking": "No parking area ahead",
    "No Stopping": "No stopping zone ahead",
    "Pedestrian Crossing": "Pedestrian crossing ahead",
    "Straight Prohibitor No Entry": "No entry ahead",
    "Barrier Ahead": "Barrier ahead"
}

# Speed limit alerts: 20–90 (step 5)
speed_limit_classes = {
    f"Speed Limit -{n}-": f"Speed limit {n} ahead, please follow"
    for n in range(20, 95, 5)
}

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

print("🎥 Detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
    unique_classes = set(detected_classes)

    # Trigger alerts
    for cls in unique_classes:
        if cls in alerts:
            speak_alert(alerts[cls])
        elif cls in speed_limit_classes:
            speak_alert(speed_limit_classes[cls])

    # Show detection output
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
try:
    os.remove("alert.mp3")
except FileNotFoundError:
    pass

