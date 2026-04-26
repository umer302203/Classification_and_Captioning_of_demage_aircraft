# AI-Powered Aircraft Damage Analysis System

## Live Demo
[Link to Hugging Face Space]()

## Overview
This is a production-grade web application that integrates two advanced AI models to perform multi-faceted analysis of aircraft damage images. The system combines a fine-tuned VGG16-based Keras model for discriminative tasks with the Salesforce BLIP model for generative tasks. Users can upload an image of aircraft damage, and the system will automatically classify the damage type, generate a descriptive caption, and provide a detailed summary—all powered by state-of-the-art deep learning models.

## Features
- **Binary Classification of Damage**: Automatically classifies aircraft damage into two categories: dents or cracks
- **Automated Image Captioning**: Generates descriptive captions for the uploaded damage image
- **Detailed Image Summarization**: Produces comprehensive summaries describing the damage characteristics
- **User-Friendly Web Interface**: Built with Gradio for seamless interaction
- **Production-Ready Architecture**: Modular, scalable design with separated concerns for easy maintenance and extension

## Technology Stack
- **Deep Learning Framework**: TensorFlow/Keras for classification
- **Generative Model**: Hugging Face Transformers (BLIP Image Captioning)
- **Web Framework**: Gradio
- **Model Hub**: Hugging Face Hub for model storage and retrieval
- **Python**: 3.8+

## Local Setup Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/aircraft_damage_app.git
   cd aircraft_damage_app
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Model Repository**
   - Open `model_handler.py`
   -repo_id = "rana789r/aircraft-damage-classifier"
   - Ensure your fine-tuned classification model is named `aircraft_damage_classifier.keras`

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **Access the Web Interface**
   - Open your browser and navigate to the URL displayed in the terminal (typically `http://127.0.0.1:7860`)
   - Upload an aircraft damage image and analyze it

## Project Architecture

### Directory Structure
```
/aircraft_damage_app/
├── app.py                    # Main Gradio application
├── model_handler.py          # Model loading utilities
├── image_processor.py        # Image preprocessing utilities
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation

```

### Module Descriptions

#### `app.py`
- **Purpose**: Core application entry point
- **Key Components**:
  - Global model initialization for optimal performance
  - `master_predict()` function orchestrating the entire pipeline
  - Gradio interface definition and launch logic

#### `model_handler.py`
- **Purpose**: Centralized model loading logic
- **Functions**:
  - `load_classification_model()`: Downloads and loads the fine-tuned VGG16 Keras model from Hugging Face Hub
  - `load_generative_model()`: Loads the BLIP model and its processor

#### `image_processor.py`
- **Purpose**: Image preprocessing utilities
- **Functions**:
  - `preprocess_for_vgg16()`: Resizes images to 224×224, rescales pixel values, and formats for batch processing

## Model Architecture

### Classification Model (VGG16-based)
- **Base Architecture**: VGG16 convolutional neural network
- **Fine-tuning**: Adapted for binary classification of aircraft damage
- **Output**: Probability score indicating damage type (crack vs. dent)
- **Input Size**: 224×224 RGB images
- **Activation**: Sigmoid for binary classification

### Generative Model (BLIP)
- **Base Model**: Salesforce/blip-image-captioning-base
- **Architecture**: Vision transformer encoder + language model decoder
- **Capabilities**: 
  - Image captioning with contextual prompts
  - Detailed image summarization
- **Pre-training**: Trained on 129M image-text pairs from diverse sources

## API Specification

### Input
- **Type**: Image (JPEG, PNG, or other common formats)
- **Resolution**: Any resolution (automatically resized to 224×224 for classification)
- **Format**: Supported by PIL (Python Imaging Library)

### Output
- **Classification Result**: String indicating damage type with confidence percentage
- **Caption**: Single-sentence description of the damage
- **Summary**: Multi-sentence detailed description of the damage characteristics

## Deployment to Hugging Face Spaces

1. Push this repository to GitHub
2. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
3. Select "Docker" as the space type
4. Connect your GitHub repository
5. The application will automatically deploy and be accessible via a public URL

## Future Enhancements
- Multi-class damage classification (expand beyond binary)
- Localization of damage regions using bounding boxes
- Damage severity quantification
- Batch processing capabilities
- Integration with maintenance record systems
- Historical comparison and trend analysis

## Contributing
Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- **VGG16**: Simonyan & Zisserman (2014) - University of Oxford
- **BLIP**: Li et al. (2022) - Salesforce Research
- **Gradio**: Hugging Face
- **Hugging Face Hub**: Model storage and distribution

## Support
For issues, questions, or suggestions, please open an issue on GitHub or contact the development team.

---

**Last Updated**: April 2026  
**Version**: 1.0.0  
**Status**: Production-Ready
