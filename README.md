# 🌱 Plant Disease Detection System Using Deep Learning

## 📌 Project Overview

The **Plant Disease Detection System** is a deep learning-based project designed to identify plant leaf diseases using image classification techniques.

The main objective of this project is to help identify unhealthy plant leaves automatically using Artificial Intelligence (AI) and Deep Learning. The system analyzes an uploaded leaf image and classifies it as Healthy or Diseased.

This project implements and compares multiple deep learning architectures to evaluate their performance in plant disease detection.

## 🎯 Project Objectives

- Automatically detect plant leaf diseases using deep learning.
- Classify plant leaves into Healthy and Diseased categories.
- Implement and compare different CNN and Vision Transformer architectures.
- Improve classification performance using attention mechanisms.
- Evaluate models using Accuracy, Precision, Recall, and F1-Score.
- Develop an image-based prediction system.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | Neural Network Development |
| NumPy | Numerical Operations |
| Scikit-learn | Model Evaluation |
| Matplotlib | Data Visualization |
| Seaborn | Performance Visualization |
| Google Colab | Model Training |
| Visual Studio Code | Development Environment |
| Git & GitHub | Version Control |

## 🧠 Deep Learning Models Implemented

The project implements and evaluates the following models:

1. Simple CNN
2. CNN with SE Attention
3. Vision Transformer (ViT)
4. Vision Transformer with Custom Self-Attention
5. VGG16
6. ResNet152
7. DenseNet
8. InceptionV4
9. EfficientNetB7
10. MobileNetV3

## 📂 Dataset Description

The project uses the PlantVillage dataset, which contains images of healthy and diseased plant leaves.

The project report describes an apple leaf subset containing healthy leaves and diseases such as:

- Apple Scab
- Black Rot
- Cedar Apple Rust

The primary classification task is binary classification:

- Healthy
- Diseased

## ⚙️ Data Preprocessing

The following preprocessing techniques are used:

- Image resizing to 224 × 224 pixels.
- Image normalization (0–1).
- Data augmentation.
- Rotation and horizontal flipping.
- Noise reduction.
- Dataset shuffling.
- Batch processing.

## 📊 Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training and Validation Accuracy
- Training and Validation Loss

The project report's Table 4.1 presents the following approximate accuracy results:

| Model | Accuracy |
|---|---:|
| Simple CNN | 93% |
| CNN + SE Attention | 94% |
| ResNet152 | 93.1% |
| InceptionV4 | 96% |
| VGG16 | 97.3% |
| DenseNet | 97.8% |
| Vision Transformer | 98% |
| ViT + Self-Attention | 99% |

*Note: These are the approximate results reported in the project report. Actual results may vary depending on training and evaluation conditions.*

## 🚀 Project Structure

```text
Plant-Disease-Detection/
│
├── DenseNet.py
├── EfficientNetB7.py
├── InceptionV4.py
├── MobileNetV3.py
├── ResNet152.py
├── VGG16_model.py
│
├── simple_cnn.py
├── plant_disease_cnn.py
├── vit_model.py
├── vit_self_model.py
│
├── predict_image.py
├── predict_DenseNet.py
├── predict_EfficientNetB7.py
├── predict_MobileNetV3.py
├── predict_ResNet152.py
├── predict_vgg16.py
├── predict_vit.py
├── predict_vit_self.py
│
├── evaluate_model.py
├── evaluate_DenseNet.py
├── evaluate_EfficientNetB7.py
├── evaluate_InceptionV4.py
├── evaluate_vit.py
│
├── Testing/
│
├── .gitignore
└── README.md
```

## 💻 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/sunny-kumar75/Plant-Disease-Detection.git
```

### Step 2: Navigate to the Project

```bash
cd Plant-Disease-Detection
```

### Step 3: Create a Virtual Environment

```bash
python3 -m venv vit_env
```

### Step 4: Activate the Environment

For macOS:

```bash
source vit_env/bin/activate
```

### Step 5: Install Required Libraries

```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

Install any additional libraries required by the individual model scripts.

### Step 6: Prepare Dataset

Organize the dataset into training and validation folders:

```text
dataset/
│
├── train/
│   ├── Healthy/
│   └── Diseased/
│
└── val/
    ├── Healthy/
    └── Diseased/
```

### Step 7: Run the Project

Run the required model training or prediction Python script using:

```bash
python3 predict_image.py
```

Make sure the corresponding trained model file and required dataset are available locally.

## 🔮 Future Enhancements

- Develop a web-based plant disease detection application.
- Integrate real-time plant disease detection.
- Support more plant species and disease categories.
- Improve model accuracy using larger datasets.
- Deploy the prediction system on cloud platforms.
- Develop a mobile application for farmers.

## 👨‍💻 Project Contributors

**Sunny Kumar**  

## 📜 License

This project is developed for academic and educational purposes.
