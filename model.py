"""
Machine Learning Model Service.
Handles model loading, image transformation, and inference for digit classification.
"""
import os
from PIL import Image
import numpy as np

try:
    from keras.models import load_model
    from keras.utils import img_to_array
except ImportError:
    load_model = None

    def img_to_array(img):
        """Converts a PIL Image instance to a Numpy array."""
        return np.asarray(img, dtype=np.float32)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.h5")
model = None

if load_model and os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception:  # pylint: disable=broad-exception-caught
        model = None


def preprocess_img(img_stream):
    """
    Resizes and normalizes an input image stream into a 4D tensor.

    Args:
        img_stream: File-like stream or path of the input image.

    Returns:
        np.ndarray: Reshaped image array ready for inference (1, 224, 224, 3).
    """
    opened_img = Image.open(img_stream)
    resized_img = opened_img.resize((224, 224))
    img_arr = img_to_array(resized_img) / 255.0
    return img_arr.reshape(1, 224, 224, 3)


def predict_result(predict_tensor):
    """
    Predicts the digit class from preprocessed image tensor.

    Args:
        predict_tensor (np.ndarray): Preprocessed image tensor.

    Returns:
        int: Predicted class index.
    """
    if model is None:
        raise RuntimeError("Machine learning model is not loaded.")
    pred = model.predict(predict_tensor)
    return int(np.argmax(pred[0], axis=-1))
