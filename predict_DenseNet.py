import tensorflow as tf
import numpy as np
import cv2
import os

# ✅ Load DenseNet model (changed)
model = tf.keras.models.load_model("densenet_model.h5")

# (Optional but useful) class labels
class_names = ["Healthy", "Diseased"]

while True:
    img_path = input("Enter image path (or type exit): ")

    if img_path.lower() == "exit":
        break

    # Check file exists
    if not os.path.exists(img_path):
        print("Image path not found ❌")
        continue

    img = cv2.imread(img_path)

    if img is None:
        print("Invalid image ❌")
        continue

    # Preprocessing
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 🔥 IMPORTANT FIX
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    prob = prediction[0][0]

    # Output
    if prob > 0.5:
        print(f"Prediction: Diseased Leaf 🍂 (Confidence: {prob:.2f})")
    else:
        print(f"Prediction: Healthy Leaf 🌿 (Confidence: {1-prob:.2f})")