import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Paths and settings
base_dir = os.getcwd()  # Current working directory (should be "Traffic Symbols Dataset")
image_size = (128, 128)
batch_size = 8
epochs = 10

# Step 2: Load dataset with ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Step 3: Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# Step 4: Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train model
model.fit(train_data, validation_data=val_data, epochs=epochs)

# Step 6: Save model and class labels
model.save("traffic_symbol_model.h5")

with open("class_labels.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("\n✅ Training Complete! Model saved as 'traffic_symbol_model.h5'")
