# 🧠 Custom Convolutional Neural Network (CNN) for Skin Image Classification

## 📌 Summary

This project implements a custom-built deep learning model capable of classifying skin images into distinct categories (e.g., acne-affected vs normal skin). Unlike solutions that rely on pre-trained black-box models, this work focuses on designing, training, and evaluating a neural network from the ground up.

The project demonstrates not only what the model predicts, but how and why those predictions are made, highlighting a strong understanding of AI fundamentals, data handling, and model evaluation.

---

## 🎯 Project Motivation

Most introductory AI projects rely heavily on pre-trained models (e.g., ResNet, MobileNet), which often hide architectural decisions, training dynamics, and feature learning behavior.

This project intentionally avoids that abstraction to:
- Demonstrate deep learning fundamentals
- Show ownership of the entire ML pipeline
- Build a model whose behavior can be analyzed and improved

The chosen application domain, skin condition analysis, represents a real-world computer vision problem with high societal impact.

---

## 🧠 Problem Definition (Technical)

**Task Type:** Binary Image Classification  
**Input:** RGB skin images  
**Output:** Class probability (skin condition category)  
**Learning Paradigm:** Supervised Learning  
**Model Type:** Custom Convolutional Neural Network (CNN)

Formally, the model learns a function:

fθ : X → Y

Where:
- X = image tensor
- Y = class label
- θ = learned parameters

---

## 🗂️ Dataset & Preprocessing Pipeline

### Dataset Characteristics
- Image-based dataset
- Balanced representation of skin conditions
- Real-world variations in texture, lighting, and contrast

### Preprocessing Steps
1. Image resizing to fixed spatial dimensions
2. Pixel normalization to stabilize gradient updates
3. Label encoding for supervised learning
4. Train–validation split to prevent data leakage

These steps ensure numerical stability and consistent feature learning.

---

## 🏗️ Model Architecture (Deep Technical Overview)

The CNN architecture follows a hierarchical feature extraction paradigm.

### 🔹 Convolutional Blocks
Each block consists of:
- Convolution layer (learnable spatial filters)
- ReLU activation
- Pooling layer for spatial downsampling

These layers progressively learn:
- Low-level features (edges, gradients)
- Mid-level textures
- High-level semantic patterns

### 🔹 Activation Function
- ReLU (Rectified Linear Unit)
- Prevents vanishing gradients
- Enables sparse activations

### 🔹 Pooling Strategy
- Reduces spatial dimensionality
- Increases translational invariance
- Lowers computational complexity

### 🔹 Fully Connected Layers
- Flattened feature maps are passed to dense layers
- Perform high-level feature combination
- Enable class separability

### 🔹 Output Layer
- Sigmoid activation
- Outputs probability score for binary classification

---

## ⚙️ Training Configuration

- Loss Function: Binary Cross-Entropy  
- Optimizer: Adaptive gradient-based optimizer  
- Learning Rate: Tuned for stable convergence  
- Epochs: Multiple passes to ensure feature refinement  
- Batch Size: Selected to balance convergence speed and stability  

Backpropagation is used to iteratively update parameters using gradient descent.

---

## 📈 Training Dynamics & Convergence

During training, the following were monitored:
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

These curves help detect overfitting, underfitting, and learning plateaus.

---

## 📊 Evaluation Metrics (Why They Matter)

| Metric     | Purpose                                   |
|------------|-------------------------------------------|
| Accuracy   | Overall correctness                        |
| Precision  | Reliability of positive predictions        |
| Recall     | Sensitivity to true positives              |
| ROC-AUC   | Threshold-independent separability         |

Using ROC-AUC is especially important for medical-adjacent classification tasks where risk sensitivity matters.

---

## 🧪 Experimental Observations

- Custom CNNs can achieve competitive performance when properly tuned
- Feature learning improves significantly across deeper layers
- Training stability depends strongly on preprocessing quality
- Validation curves indicate good generalization behavior

---

## 🧰 Technology Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

All tools used are industry-standard for AI and ML development.

---

## 📁 Repository Structure

Custom_CNN/
├── Custom_CNN.ipynb        # End-to-end CNN implementation
├── README.md              # Project documentation
└── requirements.txt       # Environment dependencies

---

## ▶️ Reproducibility

The entire experiment is reproducible using the provided notebook:
1. Clone the repository
2. Install dependencies
3. Execute the notebook sequentially

No hidden scripts or external dependencies.

---

## 🚀 Skills Demonstrated

### Machine Learning
- CNN architecture design
- Loss function selection
- Optimization strategy
- Metric-based evaluation

### Software Engineering
- Modular notebook structure
- Clean experimental workflow
- Reproducibility awareness

### Analytical Thinking
- Model diagnostics
- Performance interpretation
- Design tradeoff analysis

---

## 🔮 Future Work

Potential extensions include:
- Transfer learning comparisons
- Multi-class classification
- Explainability techniques (Grad-CAM)
- Hyperparameter optimization
- Deployment-ready inference pipelines

---

## 👤 Author

**Unmesh Achar**  
M.S. Computer Engineering  
New York University (NYU)

---

This project demonstrates not just the use of deep learning, but the ability to design, analyze, and reason about neural networks at an architectural and mathematical level.
