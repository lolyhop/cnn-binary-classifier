# CNN Binary Classifier Examples

This directory contains example scripts and notebooks for using the CNN binary classifier.

## Quick Start Example

```python
# Example: Load model and make prediction
import torch
from PIL import Image
from src.model import create_model
from torchvision import transforms

# Load the trained model
model = create_model(num_classes=2, pretrained=True)
model.eval()

# Preprocess an image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open("path/to/your/image.jpg")
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    
classes = ['Cat', 'Dog']
print(f"Prediction: {classes[predicted.item()]}")
```

## Training Example

```python
# Example: Train your own model
from train import main

# Make sure your data is organized as:
# data/train/cats/[cat images]
# data/train/dogs/[dog images]

# Run training
main()
```

## Streamlit App Usage

```bash
# Run the web interface
streamlit run app.py
```

Then upload any cat or dog image to see the prediction!

## Docker Usage

```bash
# Build and run with Docker
docker-compose up --build

# Access at http://localhost:8501
```

## Data Organization

For training, organize your data like this:

```
data/
  train/
    cats/
      cat_001.jpg
      cat_002.jpg
      ...
    dogs/
      dog_001.jpg
      dog_002.jpg
      ...
```

## Model Architecture

The CNN model includes:
- 4 Convolutional layers (32, 64, 128, 256 filters)
- Batch normalization after each conv layer
- ReLU activation functions
- Max pooling for downsampling
- Dropout for regularization (0.5)
- 3 Fully connected layers (512, 256, 2 neurons)

Input: 224x224x3 RGB images
Output: 2-class probability distribution (Cat/Dog)