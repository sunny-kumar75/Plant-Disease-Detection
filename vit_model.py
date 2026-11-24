import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vit_keras import vit

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# DATA
train_gen = ImageDataGenerator(rescale=1./255)
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

# ViT
vit_model = vit.vit_b16(
    image_size=224,
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

# ======================
# FIXED MODEL
# ======================
x = vit_model.output   # (None, 768)

print("Shape:", x.shape)

# Convert to 3D
x = tf.keras.layers.Reshape((1, 768))(x)

# Self-attention
attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = tf.keras.layers.Add()([x, attn])
x = tf.keras.layers.LayerNormalization()(x)

# FFN
ffn = tf.keras.layers.Dense(128, activation='relu')(x)
ffn = tf.keras.layers.Dense(768)(ffn)

x = tf.keras.layers.Add()([x, ffn])
x = tf.keras.layers.LayerNormalization()(x)

# Back to 2D
x = tf.keras.layers.Flatten()(x)

# Classification
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=vit_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(train_data, validation_data=val_data, epochs=5)

model.save("vit_fixed.h5")