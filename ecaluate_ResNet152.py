import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Load Model (change name if needed)
model = tf.keras.models.load_model("simple_cnn_model.h5")

val_dir = "dataset/val"

# 🔹 Preprocessing (IMPORTANT)
# 👉 For CNN → use rescale
# 👉 For ResNet → use preprocess_input

USE_RESNET = False   # ✅ Change to True if using ResNet

if USE_RESNET:
    from tensorflow.keras.applications.resnet import preprocess_input
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
else:
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 🔹 Predictions
pred_probs = model.predict(val_data)
pred = (pred_probs > 0.5).astype(int).reshape(-1)

true_labels = val_data.classes

# 🔹 Confusion Matrix
cm = confusion_matrix(true_labels, pred)
print("Confusion Matrix:\n", cm)

# 🔹 Classification Report
print("\nClassification Report:\n")
print(classification_report(true_labels, pred))

# 🔹 Accuracy
acc = accuracy_score(true_labels, pred)
print("Accuracy:", acc)

# 🔹 Precision, Recall, F1
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred, average=None)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# 🔹 Class Labels
print("\nClass Mapping:", val_data.class_indices)

# 🔹 Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()