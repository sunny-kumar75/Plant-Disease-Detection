import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Load EfficientNet model
model = tf.keras.models.load_model("efficientnetb7_model.h5")

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
pred = (pred_probs > 0.5).astype(int).flatten()

# True labels
y_true = val_data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, pred, target_names=list(val_data.class_indices.keys())))

# Accuracy
acc = accuracy_score(y_true, pred)
print("Accuracy:", acc)

# Precision, Recall, F1 Score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average=None)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=val_data.class_indices.keys(),
            yticklabels=val_data.class_indices.keys())

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - EfficientNetB7")
plt.show()