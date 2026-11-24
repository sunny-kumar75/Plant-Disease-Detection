import tensorflow as tf
import numpy as np
import cv2
from vit_keras.layers import ClassToken, AddPositionEmbs

model = tf.keras.models.load_model(
    "vit_self_attention_model.keras",
    custom_objects={
        "ClassToken": ClassToken,
        "AddPositionEmbs": AddPositionEmbs
    },
    compile=False,
    safe_mode=False
)

while True:
    img_path = input("Enter image path (or type exit): ")

    if img_path.lower() == "exit":
        break

    img = cv2.imread(img_path)

    if img is None:
        print("Image not found ❌")
        continue

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]

    print("Raw prediction:", prediction)

    # swapped class mapping
    if prediction > 0.5:
        print(f"Prediction: Healthy Leaf 🍃 ({prediction:.4f})")
    else:
        print(f"Prediction: Diseased Leaf 🌿 ({prediction:.4f})")