"""
Model Handler Module
Centralizes all model loading logic for the aircraft damage analysis system.
This module abstracts away model instantiation to keep the main application clean.
"""

import tensorflow as tf
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import hf_hub_download


def load_classification_model():
    """
    Load the fine-tuned VGG16-based Keras classification model from Hugging Face Hub.
    
    Returns:
        tf.keras.Model: The loaded classification model.
    """
    # Hugging Face Hub repository details (replace with your actual repository)
    repo_id = "YourHuggingFaceUsername/YourModelRepoName"
    filename = "aircraft_damage_classifier.keras"
    
    # Download the model file from Hugging Face Hub
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # Load the Keras model
    classification_model = tf.keras.models.load_model(model_path)
    
    return classification_model


def load_generative_model():
    """
    Load the BLIP image captioning model and its processor from Hugging Face Hub.
    
    Returns:
        tuple: (BlipProcessor, BlipForConditionalGeneration) - The processor and model objects.
    """
    # Model ID for the pre-trained BLIP model
    model_id = "Salesforce/blip-image-captioning-base"
    
    # Load the processor
    blip_processor = BlipProcessor.from_pretrained(model_id)
    
    # Load the generative model
    blip_model = BlipForConditionalGeneration.from_pretrained(model_id)
    
    return blip_processor, blip_model
