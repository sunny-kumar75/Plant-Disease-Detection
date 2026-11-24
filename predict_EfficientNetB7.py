import tensorflow as tf
import numpy as np
import cv2

# Load EfficientNet model
model = tf.keras.models.load_model("efficientnetb7_model.h5")

IMG_SIZE = (224, 224)

while True:
    img_path = input("Enter image path (or type exit): ")

    if img_path.lower() == "exit":
        break

    img = cv2.imread(img_path)

    if img is None:
        print("Image not found ❌")
        continue

    # Resize
    img = cv2.resize(img, IMG_SIZE)

    # Convert BGR → RGB (VERY IMPORTANT for pretrained models)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img / 255.0

    # Expand dims
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]

    # Print probability
    print(f"Confidence Score: {prediction:.4f}")

    if prediction > 0.5:
        print("Prediction : Diseased Leaf 🍂")
    else:
        print("Prediction : Healthy Leaf 🌿")