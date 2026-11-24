import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load model (NO ERROR now)
model = tf.keras.models.load_model("mobilenetv3_model.keras")

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
pred_probs = model.predict(val_data)
pred = (pred_probs > 0.5).astype(int).reshape(-1)

true_labels = val_data.classes

# Confusion Matrix
cm = confusion_matrix(true_labels, pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(
    true_labels,
    pred,
    target_names=list(val_data.class_indices.keys())
))

# Accuracy
print("Accuracy:", accuracy_score(true_labels, pred))

# Precision, Recall, F1
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred, average=None)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=val_data.class_indices.keys(),
            yticklabels=val_data.class_indices.keys())

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()