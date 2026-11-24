import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load DenseNet model (changed here)
model = tf.keras.models.load_model("densenet_model.h5")

val_dir = "dataset/val"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False   # VERY IMPORTANT
)

# Predictions
pred = model.predict(val_data)

# Convert probabilities → class (0/1)
pred = (pred > 0.5).astype(int).reshape(-1)

# True labels
y_true = val_data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, pred))

# Accuracy
print("Accuracy:", accuracy_score(y_true, pred))

# Precision, Recall, F1
precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average=None)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("DenseNet Confusion Matrix")
plt.show()