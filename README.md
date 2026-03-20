# 🪖 AI-Powered Smart Helmet for Road Safety

An **AI-powered smart helmet prototype** that uses **computer vision** to detect important road elements and provide **real-time voice alerts** to two-wheeler riders.

The goal of this project is to improve rider awareness and road safety using a lightweight **edge AI system** that can run on devices such as **Raspberry Pi or NVIDIA Jetson Nano**.

---

# 🏆 Achievement

🥇 **1st Prize — Hack N Tech 1.0 (24-Hour Hackathon)**
📍 Organized at **Indian Institute of Technology Patna**

Originally built during a 24-hour hackathon as part of a team (Caffeine Coderz), this system was later refined independently to explore real-world deployment challenges in edge AI systems.

I proposed the idea and led the AI system design, implementation, and hardware integration.

---

# 🎯 Problem Motivation

Two-wheeler riders often face dangerous situations due to:

* Heavy traffic
* Limited visibility
* Sudden obstacles
* Lack of assistive safety systems

This project explores how **computer vision running locally on edge hardware** can help riders detect nearby hazards and receive **real-time audio alerts**.

All processing is done **locally**, so the system works **without internet connectivity**.

---

# ⚙️ System Overview

The system captures live video from a camera mounted on the helmet and processes each frame using a **YOLOv8 object detection model**.

Detected objects are mapped to **voice alerts** to notify the rider about potential hazards.

The system emphasizes real-time decision-making under constrained environments rather than offline model accuracy benchmarks.

### Pipeline

Camera → Frame Capture → YOLOv8 Detection → Decision Logic → Voice Alert

---

## 🎥 Demo

This system performs real-time object detection and provides audio alerts for road hazards.

▶️ Watch live demo:
https://www.linkedin.com/posts/harshkumar0007_ai-machinelearning-computervision-activity-7440829619294855168-jttU

---

# ✨ Current Detection Capabilities

The current prototype uses **YOLOv8n pretrained on the COCO dataset**.

The system currently detects and alerts for:

* **Person** → Pedestrian detected ahead
* **Bicycle** → Cyclist nearby
* **Car** → Car approaching
* **Motorcycle** → Motorbike nearby
* **Bus** → Bus ahead
* **Truck** → Truck nearby
* **Traffic Light** → Traffic light ahead
* **Stop Sign** → Stop sign detected
* **Cow** → Animal on road ahead
* **Dog** → Dog detected on road

These detections trigger **audio alerts** to warn the rider.

---

# 🛠 Technologies Used

* **Python**
* **OpenCV**
* **YOLOv8 (Ultralytics)**
* **PyTorch**
* **pyttsx3 (Text-to-Speech)**

---

# 🔧 Hardware Prototype

The prototype hardware setup includes:

* Raspberry Pi (compute unit)
* Helmet-mounted camera
* PAM audio amplifier module
* Mini speaker system
* Portable battery supply

The **hardware system was fully assembled, wired, and tested by me** as part of the prototype development.

---

# 📂 Project Structure

```
ai-powered-smart-helmet-computer-vision
│
├── src
│   ├── smart_helmet_detection_system.py
│   └── detection_using_media.py
│
├── models
│   ├── yolov8n.pt
│   └── yolov8n_openvino_model
│
├── dataset
│   └── traffic_symbols_sample
│
├── experiments
│   └── prototype scripts used during development
│
├── tools
│   ├── get_names.py
│   └── test.py
│
├── docs
│
├── demo
│
├── requirements.txt
└── README.md
```

---

## 🚀 Running the Project

### 1. Clone the repository

```
git clone https://github.com/skynet-007-ai/ai-powered-smart-helmet-computer-vision.git

cd ai-powered-smart-helmet-computer-vision
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the system

```
python src/smart_helmet_detection_system.py
```

Press **q** to exit.

---

# ⚠️ Real-World Engineering Considerations

This system is an early-stage edge AI prototype built under real-time constraints.

Key challenges encountered:

- Balancing accuracy vs latency  
- Running inference on limited hardware  
- Handling noisy detections in dynamic environments  
- Designing stable alert logic without overwhelming the user  

The focus of this project is not just detection accuracy, but building a complete:

Perception → Decision → Alert pipeline under real-world constraints

---

# 🔮 Future Improvements

Planned improvements include:

* Training **custom models for traffic sign detection**
* Detecting more road-specific hazards
* Optimizing inference using **OpenVINO / TensorRT**
* Adding **vibration alerts inside the helmet**
* Building a more compact wearable hardware design

---

# 👨‍💻 Author

**Harsh Kumar**
B.Sc. (Hons.) Computer Science & Data Analytics
Indian Institute of Technology Patna

GitHub: https://github.com/skynet-007-ai

---

# 📜 License

This project is licensed under the **Apache 2.0 License**.
