import torch
import torch.nn as nn
import torch.nn.functional as F

class CatDogCNN(nn.Module):
    """
    Convolutional Neural Network for binary classification of cats and dogs.
    """
    
    def __init__(self, num_classes=2):
        super(CatDogCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # Input size calculation: 224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Fourth conv block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def create_model(num_classes=2, pretrained=False):
    """
    Create a CatDogCNN model.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for binary classification)
        pretrained (bool): Whether to load pretrained weights (default: False)
    
    Returns:
        model: CatDogCNN model instance
    """
    model = CatDogCNN(num_classes)
    
    if pretrained:
        # Load pretrained weights if available
        try:
            model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
            print("Loaded pretrained weights")
        except FileNotFoundError:
            print("No pretrained weights found, using random initialization")
    
    return model