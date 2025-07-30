import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def predict_image(model_path, image_path, image_size=(64, 64), class_indices=None):
    model = load_model(model_path)
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    if class_indices:
        labels = {v: k for k, v in class_indices.items()}
        return labels[predicted_class]
    return predicted_class

