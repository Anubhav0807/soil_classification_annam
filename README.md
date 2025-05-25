# 🌱 Soil Image Classification Challenges

This repository contains solutions for two image classification challenges organized by **Annam.ai** at **IIT Ropar**. Both tasks are centered around soil image classification but differ in their objectives and approaches.

---

## 🧠 Challenge 1: Multi-Class Soil Image Classification

### 🔍 Objective

Classify each input image into one of the following four soil categories:

- Alluvial soil
- Black soil
- Clay soil
- Red soil

### 🧪 Approach

A **Convolutional Neural Network (CNN)** built using **TensorFlow** was trained to recognize soil types.

### 🧬 Pipeline Overview

1. **Data Preparation**
    - Load image paths and labels from `train_labels.csv`
    - Map labels to integer indices

2. **Preprocessing**
    - Decode and resize images to `224x224`
    - Normalize pixel values to `[0, 1]`

3. **Model Architecture**
    - 3 Conv2D layers with ReLU + MaxPooling
    - Flatten + Dense + Softmax output layer

4. **Training**
    - Loss: Sparse Categorical Crossentropy
    - Optimizer: Adam
    - Epochs: 10

5. **Prediction and Submission**
    - Predict test images
    - Map predictions to original string labels
    - Save to `submission.csv`

---

## 🔍 Challenge 2: One-Class Soil Image Detection

### 🎯 Objective

Determine if a given image is of **soil** or **not soil** using a **one-class classification** approach.

### 🛠️ Method

A **Convolutional Autoencoder** in **PyTorch** is trained on only **soil images**.

### 🧬 Pipeline Overview

1. **Preprocessing**
    - Resize to `224x224`
    - Normalize to `[-1, 1]`

2. **Model Architecture**
    - Encoder: 3 Conv layers with ReLU + MaxPooling
    - Decoder: 3 TransposedConv layers with ReLU and final Tanh

3. **Training**
    - Loss: Mean Squared Error (MSE)
    - Optimizer: Adam
    - Epochs: 20

4. **Testing**
    - Compute reconstruction error
    - Classify as **soil (1)** if error ≤ `0.05`, otherwise **not soil (0)**

### ✅ Sample Output

```
img1.jpg      1  
img2.jpeg     0  
img3.png      1  
...
```

---

## 👥 Authors

- **Adityaraj Shrivastava**
- **Anubhav Jha**
- **Dipayan Nandi**
- **Mahargha Kundu**

---

## 📂 Folder Structure

```
.
├── challenge-1/       # Multi-class classification using CNN
├── challenge-2/       # One-class classification using Autoencoder
├── README.md          # Combined documentation
└── LICENSE            # MIT License
```

---

## 🚀 Setup & Run Instructions

### 📦 Prerequisites

Ensure you have **Python 3.8+** installed. Check with:

```bash
python --version
```

Also verify `pip` is available:

```bash
pip --version
```

---

### 🛠️ Installation

1. **Clone the Repository**

```bash
git clone https://github.com/Anubhav0807/soil_classification_annam.git
cd soil_classification_annam
```

2. **(Optional) Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### ▶️ Running Notebooks

You can open and run the notebooks using Jupyter:

```bash
pip install notebook
jupyter notebook
```

Then navigate to the notebook you want to open and run all cells.

---


## 🔮 Future Improvements

- Use pretrained networks like ResNet for better feature extraction
- Optimize anomaly threshold using validation ROC-AUC
- Explore segmentation-based anomaly detection
