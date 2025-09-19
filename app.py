import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import create_model

# Configure page
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained CNN model."""
    model_path = 'models/best_model.pth'
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return None
    
    try:
        model = create_model(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """
    Preprocess the uploaded image for model inference.
    
    Args:
        image: PIL Image object
        
    Returns:
        tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict(model, image_tensor):
    """
    Make prediction using the trained model.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        
    Returns:
        prediction: Predicted class (0: cat, 1: dog)
        confidence: Confidence score
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item()

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🐱🐶 Cat vs Dog Classifier")
    st.markdown("""
    Upload an image of a cat or dog, and our CNN model will classify it!
    
    **How it works:**
    - The model is a Convolutional Neural Network trained on cat and dog images
    - It analyzes visual features to make predictions
    - The confidence score shows how certain the model is about its prediction
    """)
    
    # Sidebar
    st.sidebar.header("Model Information")
    st.sidebar.markdown("""
    **Architecture:** Custom CNN
    - 4 Convolutional layers
    - Batch normalization
    - Max pooling
    - Dropout for regularization
    - 3 Fully connected layers
    
    **Input:** 224x224 RGB images
    **Output:** Binary classification (Cat/Dog)
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # File uploader
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image of a cat or dog for classification"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"**Image Info:**\n- Size: {image.size}\n- Mode: {image.mode}")
        
        with col2:
            st.subheader("Prediction")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                try:
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Get prediction
                    prediction, confidence = predict(model, image_tensor)
                    
                    # Display results
                    classes = ['Cat 🐱', 'Dog 🐶']
                    predicted_class = classes[prediction]
                    
                    # Show prediction with large text
                    st.markdown(f"## {predicted_class}")
                    
                    # Confidence meter
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Additional details
                    if confidence > 0.8:
                        st.success("High confidence prediction! 🎯")
                    elif confidence > 0.6:
                        st.warning("Moderate confidence prediction 🤔")
                    else:
                        st.error("Low confidence prediction ⚠️")
                    
                    # Show probabilities for both classes
                    st.subheader("Class Probabilities")
                    
                    # Get probabilities for both classes
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = F.softmax(outputs, dim=1).squeeze()
                    
                    prob_data = {
                        'Class': ['Cat 🐱', 'Dog 🐶'],
                        'Probability': [probabilities[0].item(), probabilities[1].item()]
                    }
                    
                    st.bar_chart(
                        data=prob_data,
                        x='Class',
                        y='Probability'
                    )
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    # Instructions
    st.header("Instructions")
    st.markdown("""
    1. **Upload an image:** Click the file uploader above and select an image of a cat or dog
    2. **Wait for processing:** The model will analyze the image automatically
    3. **View results:** See the prediction, confidence score, and class probabilities
    
    **Tips for better results:**
    - Use clear, well-lit images
    - Make sure the cat or dog is the main subject
    - Avoid heavily edited or filtered images
    - Higher resolution images generally work better
    """)
    
    # Example images section
    st.header("Need test images?")
    st.markdown("""
    You can find test images online or use your own photos. Here are some good sources:
    - [Unsplash](https://unsplash.com/s/photos/cat-dog)
    - [Pixabay](https://pixabay.com/images/search/pets/)
    - Your own pet photos!
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using PyTorch and Streamlit")

if __name__ == "__main__":
    main()