import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from sklearn.metrics import accuracy_score, classification_report
from src.model import create_model

def get_data_loaders(data_dir, batch_size=32, val_split=0.2):
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir (str): Path to data directory
        batch_size (int): Batch size for data loaders
        val_split (float): Fraction of data to use for validation
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transform to validation dataset
    val_dataset.dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, full_dataset.classes

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, device='cpu'):
    """
    Train the CNN model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on ('cpu' or 'cuda')
    
    Returns:
        model: Trained model
        train_history: Training history dictionary
    """
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training history
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Training on device: {device}")
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update history
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s')
        print('-' * 60)
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    return model, train_history

def save_model(model, save_path='models/best_model.pth'):
    """
    Save the trained model.
    
    Args:
        model: Trained PyTorch model
        save_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def main():
    """
    Main training function.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    data_dir = 'data/train'  # Expecting data/train/cats/ and data/train/dogs/
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found!")
        print("Please organize your data as follows:")
        print("data/")
        print("  train/")
        print("    cats/")
        print("      cat1.jpg")
        print("      cat2.jpg")
        print("      ...")
        print("    dogs/")
        print("      dog1.jpg")
        print("      dog2.jpg")
        print("      ...")
        return
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, classes = get_data_loaders(data_dir, batch_size)
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(num_classes=len(classes))
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    model, history = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
    
    # Save model
    save_model(model)
    
    print("Training completed!")

if __name__ == "__main__":
    main()