import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

# Paths
train_dir = "data/train"
model_path = "models/model.h5"

# Parameters
img_size = (224, 224)
batch_size = 32
epochs = 5

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Load existing model if available, otherwise create a new one
if os.path.exists(model_path):
    print("🔁 Loading existing model...")
    model = load_model(model_path)
else:
    print("🆕 Creating new model...")
    model = Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Retrain model
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save retrained model
model.save(model_path)
print("✅ Model retrained and saved.")
