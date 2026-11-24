import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("plant_disease_model.h5")

while True:
    img_path = input("Enter image path (or type exit): ")

    if img_path.lower() == "exit":
        break

    img = cv2.imread(img_path)

    if img is None:
        print("Image not found ❌")
        continue

    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        print("Prediction : Diseased Leaf 🍂")
    else:
        print("Prediction : Healthy Leaf 🌿")