import torch
import torch.nn as nn
import os
from src.model import create_model

def create_demo_model():
    """
    Create a demo model with random weights for testing the application.
    This allows users to test the app even without training data.
    """
    model = create_model(num_classes=2)
    
    # Initialize with random weights (already done by default)
    # This is just for demonstration purposes
    print("Created demo model with random weights")
    print("Note: This model will make random predictions until properly trained")
    
    return model

def save_demo_model():
    """
    Save a demo model for testing purposes.
    """
    model = create_demo_model()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = 'models/best_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Demo model saved to {model_path}")
    print("You can now test the Streamlit app!")

if __name__ == "__main__":
    save_demo_model()