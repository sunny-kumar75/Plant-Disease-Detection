import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from vit_keras import vit

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

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

vit_model = vit.vit_b16(
    image_size=224,
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

x = vit_model.output   # (None, 768)

print("ViT Output Shape:", x.shape)

# Convert to 3D
x = tf.keras.layers.Reshape((1, 768))(x)

attn = tf.keras.layers.MultiHeadAttention(
    num_heads=4,
    key_dim=64
)(x, x)

# Residual + Norm
x = tf.keras.layers.Add()([x, attn])
x = tf.keras.layers.LayerNormalization()(x)

# Feed Forward Network
ffn = tf.keras.layers.Dense(128, activation='relu')(x)
ffn = tf.keras.layers.Dense(768)(ffn)

x = tf.keras.layers.Add()([x, ffn])
x = tf.keras.layers.LayerNormalization()(x)

# Back to 2D
x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=vit_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model.fit(
    train_data,
    epochs=20,
  
)

#model.save("vit_self_attention_model.h5")
model.save("vit_self_attention_model.keras")

print(" Model saved successfully!")