import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, test_dir, image_size=(64, 64), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_data, test_data

