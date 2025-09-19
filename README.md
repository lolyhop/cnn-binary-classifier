# CNN Binary Classifier - Cat vs Dog

A deep learning application that classifies images of cats and dogs using a Convolutional Neural Network (CNN) built with PyTorch. The application features a user-friendly Streamlit web interface and is containerized with Docker for easy deployment.

## Features

- **Custom CNN Architecture**: Designed specifically for binary classification of cats and dogs
- **Web Interface**: Interactive Streamlit app for easy image uploads and predictions
- **Docker Support**: Containerized application for consistent deployment
- **Data Augmentation**: Comprehensive data preprocessing and augmentation for robust training
- **Model Persistence**: Save and load trained models
- **Real-time Predictions**: Fast inference with confidence scores

## Architecture

The CNN model consists of:
- 4 Convolutional layers with batch normalization
- Max pooling for spatial dimension reduction
- Dropout layers for regularization
- 3 Fully connected layers for classification
- Input: 224x224 RGB images
- Output: Binary classification (Cat/Dog) with confidence scores

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lolyhop/cnn-binary-classifier.git
   cd cnn-binary-classifier
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data** (optional, for training):
   ```
   data/
     train/
       cats/
         cat1.jpg
         cat2.jpg
         ...
       dogs/
         dog1.jpg
         dog2.jpg
         ...
   ```

3. **Train the model** (optional):
   ```bash
   python train.py
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage

### Web Interface

1. **Upload an Image**: Click the file uploader and select an image of a cat or dog
2. **View Prediction**: The model will automatically classify the image and show:
   - Predicted class (Cat 🐱 or Dog 🐶)
   - Confidence score
   - Probability distribution
3. **Interpret Results**: Higher confidence scores indicate more certain predictions

### Training Your Own Model

If you have your own dataset:

1. **Organize your data** in the following structure:
   ```
   data/
     train/
       cats/
         [cat images]
       dogs/
         [dog images]
   ```

2. **Run the training script:**
   ```bash
   python train.py
   ```

3. **Monitor training progress** - the script will show:
   - Training and validation loss
   - Training and validation accuracy
   - Best model checkpoints

## Model Performance

The model uses several techniques to achieve good performance:

- **Data Augmentation**: Random flips, rotations, and color jittering
- **Batch Normalization**: Faster training and better convergence
- **Dropout**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate for optimal training
- **Early Stopping**: Saves the best performing model

## File Structure

```
cnn-binary-classifier/
├── app.py                 # Streamlit web application
├── train.py              # Model training script
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker container configuration
├── docker-compose.yml   # Docker Compose setup
├── src/
│   └── model.py         # CNN model architecture
├── models/              # Saved model weights
├── data/                # Training data (not included)
└── README.md           # This file
```

## Requirements

- Python 3.9+
- PyTorch 2.1.0+
- Streamlit 1.28.1+
- PIL (Pillow)
- NumPy
- Matplotlib
- scikit-learn

## Tips for Best Results

- **Image Quality**: Use clear, well-lit images
- **Subject Focus**: Ensure the cat or dog is the main subject
- **Resolution**: Higher resolution images generally work better
- **Variety**: Test with different breeds, poses, and backgrounds

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Streamlit team for the user-friendly web app framework
- The open-source community for inspiration and resources