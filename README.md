🌱 Plant Disease Detection using Deep Learning & YOLOv8
📌 Introduction

Plant diseases are one of the biggest challenges in agriculture because they directly affect crop quality and productivity. Early disease detection can help farmers reduce losses and improve crop management.

This project presents an AI-based solution for detecting and classifying plant diseases using Deep Learning and Computer Vision techniques.

The project combines:

CNN (Convolutional Neural Networks) for image classification

YOLOv8 for object detection and localization

The system analyzes plant leaf images, predicts the disease type, and detects infected regions automatically.

🎯 Project Objectives

The main objectives of this project are:

Detect plant diseases automatically from leaf images

Improve disease diagnosis accuracy using Deep Learning

Reduce manual inspection effort in agriculture

Apply image preprocessing and augmentation techniques

Train and evaluate CNN and YOLOv8 models

Visualize model performance using graphs and metrics

🧠 Deep Learning Concepts Used

This project includes several important AI and Deep Learning concepts such as:

🔹 Convolutional Neural Networks (CNN)

CNN is used for extracting features from leaf images and classifying diseases.

🔹 YOLOv8 Object Detection

YOLOv8 is used to:

Detect infected areas

Draw bounding boxes around diseases

Localize disease regions in the image

🔹 Data Augmentation

Used to improve generalization and reduce overfitting through:

Rotation

Zoom

Horizontal flipping

Rescaling

🔹 Batch Normalization

Helps stabilize and speed up training.

🔹 Dropout

Used to reduce overfitting by randomly disabling neurons during training.

🔹 Optimizers

The project uses optimizers like:

Adam
SGD
📂 Dataset

The dataset contains plant leaf images for:

Healthy plants

Diseased plants

The images are divided into:

Training set

Validation set

Testing set

The dataset is preprocessed before training to improve model performance.

⚙️ Data Preprocessing

Several preprocessing techniques are applied before training:

✅ Image Resizing

All images are resized to a fixed size to fit the neural network input.

✅ Normalization

Pixel values are normalized between 0 and 1.

✅ Data Augmentation

Additional modified images are generated to improve model robustness.

🛠️ Technologies & Libraries Used

💻 Programming Language

Python

📚 Libraries

TensorFlow

Keras

PyTorch

OpenCV

NumPy

Pandas

Matplotlib

Scikit-learn

Ultralytics YOLOv8

☁️ Development Environment

Google Colab

Jupyter Notebook

🧱 CNN Model Architecture

The CNN architecture includes:

Convolution Layers

ReLU Activation Function

MaxPooling Layers

Batch Normalization

Dropout Layers

Fully Connected Dense Layers

Softmax Output Layer

The model is trained using:

Categorical Crossentropy Loss

Adam Optimizer

🚀 Project Workflow

Step 1 — Import Libraries

Necessary libraries for Deep Learning and image processing are imported.

Step 2 — Load Dataset

The dataset is loaded from directories.

Step 3 — Preprocess Images

Images are resized, normalized, and augmented.

Step 4 — Split Dataset

The dataset is divided into:

Training Data

Validation Data

Testing Data

Step 5 — Build CNN Model

The CNN architecture is created using Keras.

Step 6 — Train the Model

The model is trained on plant disease images.

Step 7 — Evaluate Performance

Accuracy and loss are calculated.

Step 8 — Visualize Results

Graphs for training and validation performance are plotted.

Step 9 — YOLOv8 Detection

YOLOv8 is trained to detect infected regions on leaves.

📊 Model Evaluation

The project evaluates performance using:

✅ Accuracy

Measures correct predictions percentage.

✅ Loss

Measures prediction error during training.

✅ Validation Accuracy

Shows model performance on unseen data.

✅ Detection Results

Displays bounding boxes around infected areas.

📈 Visualization

The project generates several graphs including:

Training Accuracy Graph

Validation Accuracy Graph

Training Loss Graph

Validation Loss Graph

These graphs help analyze:

Model learning progress

Overfitting

Underfitting

🌿 YOLOv8 Detection

YOLOv8 is used for:

Disease localization

Bounding box generation

Real-time object detection

The detection model predicts:

Disease class

Confidence score

Coordinates of infected areas

📸 Sample Output

The final output includes:

Predicted disease type

Confidence score

Bounding boxes around infected regions

Dataset Link:

https://universe.roboflow.com/plant-disease-qn8xm/plant-disease-detection-4htzn

Open:

PR_PROJECT.ipynb

📁 Project Structure


plant-disease-detection/

│

├── dataset/

├── train/

├── valid/

├── test/

├── models/

├── results/

├── PR_PROJECT.ipynb

├── requirements.txt

└── README.md

🌍 Applications

This project can be used in:

Smart Agriculture

Precision Farming

Agricultural Monitoring Systems

Automated Crop Inspection

Results

Precision increased from 97.10% to 97.49%

Recall improved from 97.20% to 97.97%

mAP@50 increased from 98% to 98.97%

mAP@50-95 improved from 71.60% to 74.17%
