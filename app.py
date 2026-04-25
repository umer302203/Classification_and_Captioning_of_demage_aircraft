"""
Aircraft Damage Analysis Web Application
A production-grade Gradio application integrating VGG16 classification
and BLIP image captioning for aircraft damage analysis.
"""

import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

from model_handler import load_classification_model, load_generative_model
from image_processor import preprocess_for_vgg16


# ============================================================================
# Global Model Initialization (Critical Performance Optimization)
# ============================================================================
# Load all models into memory upon application startup
CLASSIFICATION_MODEL = load_classification_model()
BLIP_PROCESSOR, BLIP_MODEL = load_generative_model()

# Class labels for the classification model
CLASS_NAMES = ['crack', 'dent']


# ============================================================================
# Master Prediction Function
# ============================================================================
def master_predict(uploaded_image):
    """
    Orchestrate the entire AI pipeline for aircraft damage analysis.
    
    Performs:
    1. Binary classification of damage type (crack or dent)
    2. Caption generation for the damage
    3. Detailed summary generation
    
    Args:
        uploaded_image (np.ndarray): Input image as a NumPy array from Gradio.
    
    Returns:
        tuple: (classification_label, caption_string, summary_string)
    """
    
    # ========================================================================
    # Sub-task A: Classification
    # ========================================================================
    # Preprocess image for VGG16 model
    preprocessed_image = preprocess_for_vgg16(uploaded_image)
    
    # Get prediction from classification model
    prediction = CLASSIFICATION_MODEL.predict(preprocessed_image)
    
    # Decode prediction using 0.5 threshold
    # prediction is a probability between 0 and 1
    probability = prediction[0][0]
    classification_label = CLASS_NAMES[0] if probability < 0.5 else CLASS_NAMES[1]
    confidence = (1 - probability) if probability < 0.5 else probability
    
    classification_result = f"{classification_label.upper()} (Confidence: {confidence:.2%})"
    
    # ========================================================================
    # Sub-task B: Generation (Caption and Summary)
    # ========================================================================
    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(uploaded_image.astype('uint8'))
    
    # Caption Generation
    caption_prompt = "a photo of"
    caption_inputs = BLIP_PROCESSOR(pil_image, caption_prompt, return_tensors="pt")
    caption_output_ids = BLIP_MODEL.generate(**caption_inputs, max_length=50)
    caption_string = BLIP_PROCESSOR.decode(caption_output_ids[0], skip_special_tokens=True)
    
    # Summary Generation
    summary_prompt = "a detailed description of the image is:"
    summary_inputs = BLIP_PROCESSOR(pil_image, summary_prompt, return_tensors="pt")
    summary_output_ids = BLIP_MODEL.generate(**summary_inputs, max_length=100)
    summary_string = BLIP_PROCESSOR.decode(summary_output_ids[0], skip_special_tokens=True)
    
    return classification_result, caption_string, summary_string


# ============================================================================
# Gradio Interface Definition
# ============================================================================
interface = gr.Interface(
    fn=master_predict,
    inputs=gr.Image(type="numpy", label="Upload Aircraft Damage Image"),
    outputs=[
        gr.Textbox(label="Classification Result"),
        gr.Textbox(label="Generated Caption"),
        gr.Textbox(label="Generated Summary", lines=5)
    ],
    title="AI-Powered Aircraft Damage Analysis System",
    description="This application classifies aircraft damage (dent/crack) and generates a detailed caption and summary using a hybrid VGG16 and BLIP model architecture.",
    allow_flagging="never",
    examples=None  # Update with actual sample image path if available
)


# ============================================================================
# Application Launch
# ============================================================================
if __name__ == "__main__":
    interface.launch()
