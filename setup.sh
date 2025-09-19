#!/bin/bash

# CNN Binary Classifier Setup Script
# This script helps set up the development environment

echo "🐱🐶 CNN Binary Classifier Setup"
echo "================================"

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment (optional)
read -p "Create a virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "To activate: source venv/bin/activate"
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p models data/train/cats data/train/dogs

# Create demo model
echo "Creating demo model..."
python3 create_demo_model.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your training data to data/train/cats/ and data/train/dogs/"
echo "2. Train your model: python3 train.py"
echo "3. Run the Streamlit app: streamlit run app.py"
echo "4. Or use Docker: docker-compose up --build"