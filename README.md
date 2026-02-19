# 🚀 AI_Activities

This repository contains structured Applied Artificial Intelligence activities completed as part of COMP 9130 (Applied AI).

The activities demonstrate progression from classical machine learning to neural networks and computer vision, emphasizing implementation, evaluation, and interpretability.

---

# 🎓 AI Activities (COMP 9130)

---

## 🧮 Activity I – Regression & Gradient Descent

**Dataset:** California Housing Dataset  
**Focus:** Supervised Learning – Regression  

Key implementations:
- Linear Regression from scratch using NumPy
- Manual gradient descent (forward pass, loss, gradients, parameter updates)
- Model evaluation (RMSE, MAE, R²)
- Comparison of Linear, Ridge, and Lasso regression
- Feature standardization and performance comparison

Highlights:
- Achieved 87% loss reduction during gradient descent training
- Compared regularized vs non-regularized models
- Visualized loss curves and regression fit

---

## 📊 Activity II – Credit Card Fraud Detection

**Dataset:** 284,807 transactions (0.17% fraud)  
**Focus:** Imbalanced Classification  

Key implementations:
- Logistic Regression
- Decision Tree
- Random Forest with GridSearchCV
- Confusion matrices and ROC curves
- Precision, Recall, F1-score analysis
- Handling severe imbalance using:
  - `class_weight`
  - SMOTE (Synthetic Minority Oversampling)

Highlights:
- Evaluated models beyond accuracy (fraud recall emphasis)
- Maintained stratified splits
- Demonstrated practical imbalanced data strategies

---

## 🧠 Activity III – Customer Segmentation (Unsupervised Learning)

**Dataset:** UCI Wholesale Customers (440 customers)  
**Focus:** Clustering & Dimensionality Reduction  

Key implementations:
- K-Means clustering
- Optimal K selection using:
  - Elbow method
  - Silhouette score
- PCA and t-SNE visualization
- UMAP (optional)
- Anomaly detection using Isolation Forest
- Business interpretation of cluster results

Highlights:
- Standardized features before clustering
- Compared multiple dimensionality reduction techniques
- Interpreted clusters in real-world business context

---

## 🔢 Activity IV – Neural Networks Fundamentals

**Dataset:** MNIST (60,000 images)  
**Focus:** Neural Networks from Scratch & Keras MLP  

Key implementations:
- Forward propagation implemented manually using NumPy
- Custom ReLU and Softmax functions
- 2-layer neural network (784 → 128 → 10)
- Parameter initialization and shape validation
- Multi-Layer Perceptron (MLP) using Keras
- Activation experiments & regularization
- Training curve analysis (overfitting vs underfitting)

Highlights:
- Implemented 100,000+ parameter network manually
- Diagnosed training dynamics
- Compared manual implementation with Keras pipeline

---

## 🔥 Activity V – CNN for Wildfire Detection (Colab)

**Dataset:** ~1,900 wildfire images (binary: fire vs no-fire)  
**Focus:** Convolutional Neural Networks  

Key implementations:
- Built CNN from scratch using Conv2D and pooling layers
- Binary image classification
- Keras data augmentation
- Feature map and filter visualization
- GPU-based training (T4 GPU)
- Optimized TensorFlow data pipeline

Highlights:
- ~1,800 training images
- Applied augmentation to reduce overfitting
- Visualized CNN internal feature representations
- Demonstrated understanding of convolutional weight sharing

---

# 🧠 Technical Themes Across Activities

- From-scratch implementations (gradient descent, NN forward pass)
- Proper evaluation beyond accuracy
- Handling imbalanced datasets
- Regularization (Ridge, Lasso, NN techniques)
- Dimensionality reduction (PCA, t-SNE)
- CNN architecture fundamentals
- GPU acceleration workflows

---

# 🛠 Installation

```bash
git clone https://github.com/your-username/AI_Activities.git
cd AI_Activities
pip install -r requirements.txt
