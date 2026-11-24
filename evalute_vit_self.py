import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import keras
from vit_keras.layers import ClassToken, AddPositionEmbs

model = keras.models.load_model(
    "vit_self_attention_model.keras",
    custom_objects={
        "ClassToken": ClassToken,
        "AddPositionEmbs": AddPositionEmbs
    },
    compile=False,
    safe_mode=False
)

val_dir = "dataset/val"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

pred = model.predict(val_data)
pred = (pred > 0.5).astype(int).flatten()

y_true = val_data.classes

cm = confusion_matrix(y_true, pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:\n")
print(classification_report(y_true, pred))

acc = accuracy_score(y_true, pred)
print("Accuracy ", acc)

