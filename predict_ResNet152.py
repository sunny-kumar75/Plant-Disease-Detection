import tensorflow as tf
import numpy as np
import cv2

# ✅ Load ResNet model
model = tf.keras.models.load_model("resnet152_model.h5")

# ✅ Import ResNet preprocessing
from tensorflow.keras.applications.resnet import preprocess_input

while True:
    img_path = input("Enter image path (or type exit): ")

    if img_path.lower() == "exit":
        break

    img = cv2.imread(img_path)

    if img is None:
        print("Image not found ❌")
        continue

    # Resize
    img = cv2.resize(img, (224, 224))

    # Convert BGR → RGB (IMPORTANT for ResNet)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to array
    img = np.array(img)

    # Apply ResNet preprocessing
    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    prob = prediction[0][0]

    print(f"Confidence: {prob:.4f}")

    if prob > 0.5:
        print("Prediction : Diseased Leaf 🍂")
    else:
        print("Prediction : Healthy Leaf 🌿")