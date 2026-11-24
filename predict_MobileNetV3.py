import tensorflow as tf
import numpy as np
import cv2

# ✅ Load NEW model (no .h5)
model = tf.keras.models.load_model("mobilenetv3_model.keras")

IMG_SIZE = (224, 224)

while True:
    img_path = input("Enter image path (or type exit): ")

    if img_path.lower() == "exit":
        break

    img = cv2.imread(img_path)

    if img is None:
        print("Image not found ❌")
        continue

    # ✅ Resize
    img = cv2.resize(img, IMG_SIZE)

    # ⚠️ IMPORTANT (MobileNetV3 preprocessing)
    img = img / 255.0   # keep this (since you trained with rescale=1./255)

    # Expand dims
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0][0]

    print(f"Raw Prediction Score: {prediction:.4f}")

    # ✅ Correct label mapping (IMPORTANT)
    if prediction > 0.5:
        print("Prediction: Healthy Leaf 🌿")
    else:
        print("Prediction: Diseased Leaf 🍂")