import tensorflow as tf
import numpy as np
import cv2

# ======================
# LOAD VGG16 MODEL
# ======================
model = tf.keras.models.load_model("vgg16_self_attention_model.keras")
# OR if you saved .h5
# model = tf.keras.models.load_model("vgg16_model.h5")

# ======================
# CLASS LABELS (VERY IMPORTANT)
# ======================
class_names = ["Healthy 🌿", "Diseased 🍂"]   # order must match training

# ======================
# PREDICTION LOOP
# ======================
while True:
    img_path = input("\nEnter image path (or type exit): ")

    if img_path.lower() == "exit":
        break

    img = cv2.imread(img_path)

    if img is None:
        print("❌ Image not found")
        continue
 
    # Preprocessing
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    confidence = prediction[0][0]

    # Binary classification
    if confidence > 0.5:
        label = class_names[1]
    else:
        label = class_names[0]

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")