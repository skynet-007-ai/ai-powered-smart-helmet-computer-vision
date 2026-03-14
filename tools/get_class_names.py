from ultralytics import YOLO

# Load the model you are using
model = YOLO('yolov8n.pt')

# The names are stored in a dictionary called model.names
# Let's print them out nicely
print("--- All objects the yolov8n.pt model can detect ---")
for class_id, class_name in model.names.items():
    print(f"ID: {class_id}, Name: '{class_name}'")

print("\n--- You can copy and paste these names into your alerts dictionary! ---")