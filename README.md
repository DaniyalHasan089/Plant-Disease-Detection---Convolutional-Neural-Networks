# 🌿 Plant Disease Detection Using Convolutional Neural Networks

A comprehensive deep learning solution for detecting and classifying plant diseases from leaf images. This project leverages Convolutional Neural Networks (CNNs) to identify 38 different plant disease categories with high accuracy.

## 🚀 Features

- **Multi-Class Classification**: Detects 38 different plant disease categories
- **High Accuracy**: Achieves 83.25% test accuracy on the validation set
- **Multiple Deployment Options**: 
  - Gradio web interface for easy testing
  - Flask web application with ngrok tunneling
  - TensorFlow Lite model for mobile deployment
- **Comprehensive Dataset**: Uses the New Plant Diseases Dataset (Augmented) with 70,295 training images
- **Real-time Prediction**: Instant disease detection from uploaded images
- **Cross-Platform**: Compatible with Google Colab, local environments, and cloud deployment

## 📊 Supported Plant Categories

The model can detect diseases in the following plants:
- **Apple**: Cedar Rust, Black Rot, Scab, Healthy
- **Cherry**: Powdery Mildew, Healthy
- **Corn**: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- **Grape**: Black Rot, Esca, Leaf Blight, Healthy
- **Peach**: Bacterial Spot, Healthy
- **Pepper**: Bacterial Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Strawberry**: Leaf Scorch, Healthy
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

## 🏗️ Architecture

### CNN Model Structure
```
Input Layer: (256, 256, 3)
├── Conv2D (32 filters, 3x3, ReLU) + Conv2D (32 filters, 3x3, ReLU)
├── MaxPooling2D (3x3)
├── Conv2D (64 filters, 3x3, ReLU) + Conv2D (64 filters, 3x3, ReLU)
├── MaxPooling2D (3x3)
├── Conv2D (128 filters, 3x3, ReLU) + Conv2D (128 filters, 3x3, ReLU)
├── MaxPooling2D (3x3)
├── Conv2D (256 filters, 3x3, ReLU) + Conv2D (256 filters, 3x3, ReLU)
├── Conv2D (512 filters, 5x5, ReLU) + Conv2D (512 filters, 5x5, ReLU)
├── Flatten
├── Dense (1568 units, ReLU) + Dropout (0.5)
└── Output Layer: Dense (38 units, Softmax)
```

### Model Specifications
- **Total Parameters**: ~4.5M
- **Input Size**: 256x256x3 RGB images
- **Output Classes**: 38
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Activation**: ReLU (hidden layers), Softmax (output)

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 54.30% |
| **Test Accuracy** | 83.25% |
| **Precision Score** | 83.25% |
| **Recall Score** | 83.25% |

*Note: Training accuracy is lower due to single epoch training. Multiple epochs would improve this significantly.*

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Google Colab (recommended for initial setup)

### Quick Start (Google Colab)

1. **Clone the repository**
   ```bash
   !git clone https://github.com/yourusername/plant-disease-detection-cnn.git
   %cd plant-disease-detection-cnn
   ```

2. **Install required packages**
   ```bash
   !pip install -q kaggle gradio flask flask-ngrok pyngrok
   ```

3. **Download the dataset**
   ```bash
   !kaggle datasets download -d vipoooool/new-plant-diseases-dataset
   !unzip new-plant-diseases-dataset.zip
   ```

4. **Run the notebook**
   - Open `Plant_Disease_Detection_Using_Convolutional_Neural_Network.ipynb`
   - Execute all cells sequentially

### Local Environment Setup

1. **Create virtual environment**
   ```bash
   python -m venv plant-disease-env
   source plant-disease-env/bin/activate  # On Windows: plant-disease-env\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset and run**
   ```bash
   python download_dataset.py
   python train_model.py
   ```

## 🚀 Usage

### 1. Training the Model

```python
# Load and preprocess data
train_gen = image_dataset_from_directory(
    directory="path/to/train",
    image_size=(256, 256)
)

# Train the model
history = model.fit_generator(
    train_gen,
    validation_data=test_gen,
    epochs=10
)

# Save the model
model.save("plant_disease.h5")
```

### 2. Making Predictions

```python
# Load the trained model
model = tf.keras.models.load_model('plant_disease.h5')

# Preprocess image
img = image.load_img('leaf_image.jpg', target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Predict
prediction = model.predict(img_array)
class_label = np.argmax(prediction)
```

### 3. Gradio Interface

```python
import gradio as gr

def predict_disease(img_file):
    # Your prediction logic here
    return prediction_result

iface = gr.Interface(
    fn=predict_disease, 
    inputs="file", 
    outputs=["image", "text"], 
    live=True
)
iface.launch()
```

### 4. Flask Web Application

```bash
# Run the Flask app
python app.py

# Access via ngrok tunnel
# The app will be available at the provided ngrok URL
```

## 📁 Project Structure

```
Plant Disease Detection - CNN/
├── Plant_Disease_Detection_Using_Convolutional_Neural_Network.ipynb  # Main notebook
├── README.md                                                         # This file
├── requirements.txt                                                  # Python dependencies
├── models/                                                          # Saved models
│   ├── plant_disease.h5                                            # Keras model
│   └── model.tflite                                                # TensorFlow Lite model
├── static/                                                          # Web app assets
│   ├── css/
│   ├── js/
│   └── images/
├── templates/                                                       # HTML templates
│   ├── Home.html
│   └── Plant-Disease-Detector.html
└── app.py                                                          # Flask application
```

## 🔧 Configuration

### Model Parameters
- **Image Size**: 256x256 pixels
- **Batch Size**: Default (32)
- **Learning Rate**: 0.0001
- **Epochs**: Configurable (default: 1 for quick testing)

### Dataset Configuration
- **Training Images**: 70,295
- **Validation Images**: 17,572
- **Classes**: 38
- **Image Format**: JPG
- **Augmentation**: Pre-applied in dataset

## 🌐 Deployment Options

### 1. Google Colab
- Ideal for development and testing
- Free GPU access
- Easy sharing and collaboration

### 2. Local Environment
- Full control over resources
- Customizable configurations
- Production-ready setup

### 3. Cloud Deployment
- Scalable infrastructure
- API endpoints
- Batch processing capabilities

### 4. Mobile Deployment
- TensorFlow Lite conversion
- Edge device compatibility
- Offline inference

## 📊 Dataset Information

The project uses the **New Plant Diseases Dataset (Augmented)** from Kaggle:
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **License**: Copyright Authors
- **Size**: ~2.7GB
- **Images**: 87,867 total
- **Categories**: 38 classes (healthy + diseased states)

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Areas
- Model architecture improvements
- Additional plant species support
- Performance optimization
- Documentation enhancements
- Bug fixes and testing

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) by vipoooool
- **Frameworks**: TensorFlow, Keras, Gradio, Flask
- **Platform**: Google Colab for development and testing

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername]
- **LinkedIn**: [Your LinkedIn Profile]

## 🔮 Future Enhancements

- [ ] Transfer learning with pre-trained models (ResNet, EfficientNet)
- [ ] Data augmentation techniques
- [ ] Model ensemble methods
- [ ] Real-time video processing
- [ ] Mobile app development
- [ ] API documentation
- [ ] Docker containerization
- [ ] CI/CD pipeline setup

---

⭐ **Star this repository if you find it helpful!**

🌱 **Help us grow by contributing to this project!** 