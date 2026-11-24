import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# LOAD VGG16 MODEL
# ======================
model = tf.keras.models.load_model("vgg16_self_attention_model.keras")
# OR if you saved .h5
# model = tf.keras.models.load_model("vgg16_model.h5")

# ======================
# DATA
# ======================
val_dir = "dataset/val"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ======================
# PREDICTION
# ======================
pred = model.predict(val_data)

# Convert probabilities to 0/1
pred = (pred > 0.5).astype(int).flatten()

# ======================
# METRICS
# ======================
cm = confusion_matrix(val_data.classes, pred)

print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(val_data.classes, pred))

precision, recall, f1, _ = precision_recall_fscore_support(
    val_data.classes, pred, average=None
)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nAccuracy:", accuracy_score(val_data.classes, pred))

# ======================
# HEATMAP
# ======================
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()