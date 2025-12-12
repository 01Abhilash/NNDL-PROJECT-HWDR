Handwritten Digit Recognition — MNIST PNG (CNN + Pygame Application)

A complete Deep Learning pipeline for digit classification and real-time handwritten digit recognition.

📌 Overview

This project implements a Convolutional Neural Network (CNN) trained on the MNIST PNG dataset to recognize handwritten digits (0–9).
It contains:

⭐ Model Training (CNN optimized for PNG images)

⭐ Evaluation (Confusion matrices, PR curves, classification metrics)

⭐ Fully Interactive Pygame-based Digit Recognition App

⭐ End-to-end dataset → model → deployment workflow

This project is suitable for:

Neural Networks & Deep Learning (NNDL) course projects

GitHub portfolio

Demonstrations of model deployment with interactive UI

📂 Dataset

The project uses the MNIST dataset converted to PNG format:

🔗 Dataset Download (Kaggle – AlexanderYYY MNIST PNG)

https://www.kaggle.com/datasets/alexanderyyy/mnist-png/data

Dataset structure expected:

mnist_png/
   train/
      0/
      1/
      ...
      9/
   test/
      0/
      1/
      ...
      9/


Place the dataset in:

D:/NNDL PROJECT 2/archive/mnist_png


Or update the dataset path inside:

training_code.ipynb

evaluation.py

📁 Project Structure
NNDL PROJECT 2/
│
├── archive/
│   └── mnist_png/                  # Kaggle dataset (PNG format)
│
├── saved_images/                   # Auto-saved Pygame screenshots & processed uploads
│
├── application.py                  # Pygame interface for drawing & uploading digits
├── training_code.ipynb             # CNN training pipeline
├── evaluation.py                   # Performance analysis & plots
│
├── bestmodel_png.h5                # Best saved model
│
├── confusion_matrices.png          # Train + Test confusion matrices
├── precision_recall_curve.png      # PR curve for all 10 digits
│
├── screen1.png – screen5.png       # Screens captured from app
│
└── README.md                       # (This file)

⭐ Features
🧠 Deep Learning (Training Phase)

CNN model with:

Batch Normalization

Dropout regularization

Adam optimizer

Data augmentation:

Rotation, zoom, shear

Width/height shift

Automatic LR reduction

ModelCheckpoint saves best-performing model

📊 Evaluation (Testing Phase)

Generates:

✔ Training Confusion Matrix

✔ Testing Confusion Matrix

✔ Precision–Recall Curve per digit (0–9)

✔ Classification metrics

Accuracy

Precision

Recall

F1-score

✏️ Real-Time Application (Deployment Phase)

An interactive Pygame application that allows:

Drawing digits using mouse

Uploading images containing digits

Automatic digit detection (contours)

Prediction with confidence percentage

Saving processed screens for documentation

⚙️ Installation
1️⃣ Install Python

Use Python 3.9 – 3.11 for TensorFlow compatibility.

Download:
https://www.python.org/downloads/

2️⃣ Install Dependencies

(optional) Create a virtual environment:

python -m venv venv
venv\Scripts\activate     # Windows


Install required packages:

pip install -r requirements.txt


Sample requirements.txt:

tensorflow>=2.10
numpy
matplotlib
opencv-python
pygame
scikit-learn

⚠ TensorFlow Windows Import Error (DLL Issue)

If you get this:

ImportError: DLL load failed while importing _pywrap_tensorflow_internal


Follow official help:
📌 https://www.tensorflow.org/install/pip#windows

Ensure you have:

Correct Python version

Microsoft Visual C++ Redistributable installed

CUDA/cuDNN (only for GPU builds)

🚀 Running the Project
▶️ 1. Train the Model

Open training_code.ipynb and run all cells.

Or:

python training_code.py


Model saved as:

bestmodel_png.h5

🧪 2. Evaluate the Model
python evaluation.py


Output files:

confusion_matrices.png

precision_recall_curve.png

This script provides all performance metrics.

🎮 3. Start the Digit Recognition App
python application.py

Controls
Key	Action
ENTER	Continue / Proceed
1	Drawing Mode
2	Upload Image Mode
S	Save screenshot
C	Clear board
BACKSPACE	Go back
Q	Quit
🧠 Model Architecture

Input: 28×28 grayscale PNG

Conv2D → BN → Conv2D → BN → MaxPool → Dropout

Conv2D → BN → Conv2D → BN → MaxPool → Dropout

Flatten → Dense → BN → Dropout

Output: Softmax (10 classes)

Trained using Adam optimizer for 10 epochs with augmentation.

📊 Generated Evaluation Outputs
🔹 Confusion Matrices (Train + Test)

File: confusion_matrices.png
Displays correct vs incorrect classifications.

🔹 Precision–Recall Curve

File: precision_recall_curve.png
Shows PR performance for each digit: 0–9.

🧰 Troubleshooting
❌ Pygame window not showing

Install/update pygame:

pip install pygame --upgrade

❌ Model predictions incorrect

Ensure correct dataset directory

Retrain model

Avoid altering preprocessing pipeline

❌ TensorFlow import error

Verify your Python version and Windows TensorFlow prerequisites.

📌 Future Enhancements

Deploy model via Flask / FastAPI

Add webcam-based real-time digit detection

Export mobile version (TFLite)

Deeper CNN or ResNet-style architecture

🧑‍💻 Author

Abhilash K R
Department of CSE (AI),
Ballari Institute of Technology & Management
USN: 3BR22CA001
