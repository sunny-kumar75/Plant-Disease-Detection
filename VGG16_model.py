import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16

# ======================
# PARAMETERS
# ======================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================
# DATA GENERATORS
# ======================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    "dataset/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ======================
# LOAD VGG16
# ======================
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# ======================
# MODEL BUILD
# ======================
x = base_model.output   # (None, 7, 7, 512)

# Convert to sequence
x = tf.keras.layers.Reshape((49, 512))(x)

# ======================
# SELF ATTENTION
# ======================
attn = tf.keras.layers.MultiHeadAttention(
    num_heads=4,
    key_dim=64
)(x, x)

x = tf.keras.layers.Add()([x, attn])
x = tf.keras.layers.LayerNormalization()(x)

# Feed Forward
ffn = tf.keras.layers.Dense(256, activation='relu')(x)
ffn = tf.keras.layers.Dense(512)(ffn)

x = tf.keras.layers.Add()([x, ffn])
x = tf.keras.layers.LayerNormalization()(x)

# ======================
# CLASSIFIER
# ======================
x = tf.keras.layers.GlobalAveragePooling1D()(x)

x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# ======================
# FINAL MODEL
# ======================
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# ======================
# COMPILE
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# EARLY STOPPING
# ======================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ======================
# TRAIN
# ======================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop]
)

# ======================
# SAVE MODEL
# ======================
model.save("vgg16_self_attention_model.keras")

print("Model saved successfully!")