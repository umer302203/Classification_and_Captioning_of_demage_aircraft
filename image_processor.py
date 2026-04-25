"""
Image Processor Module
Consolidates all image preprocessing logic for the aircraft damage analysis system.
Handles image transformations required by VGG16 and BLIP models.
"""

import tensorflow as tf
import numpy as np


def preprocess_for_vgg16(image_np):
    """
    Preprocess an image for VGG16 classification model.
    
    Args:
        image_np (np.ndarray): Input image as a NumPy array.
    
    Returns:
        tf.Tensor: Preprocessed image tensor with shape (1, 224, 224, 3).
    """
    # Resize image to VGG16 input size (224, 224)
    resized_image = tf.image.resize(image_np, (224, 224))
    
    # Rescale pixel values to [0, 1] range
    rescaled_image = resized_image / 255.0
    
    # Expand dimensions to create batch of one: (1, 224, 224, 3)
    batched_image = tf.expand_dims(rescaled_image, axis=0)
    
    return batched_image
