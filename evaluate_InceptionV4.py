import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load Inception model
model = tf.keras.models.load_model("inception_model.h5")

val_dir = "dataset/val"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Predictions
pred = model.predict(val_data)
pred = (pred > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(val_data.classes, pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(val_data.classes, pred))

# Accuracy
acc = accuracy_score(val_data.classes, pred)
print("Accuracy:", acc)

# 🔥 Heatmap visualization
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Inception Model")
plt.show()