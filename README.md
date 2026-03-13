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

## 🌸 Activity VI – Transfer Learning with Pretrained CNNs (Colab)

**Dataset:** Animals-10 subset (~1,000 images; 5 classes: cat, cow, dog, horse, sheep)  
**Focus:** Transfer Learning & Fine-Tuning Strategies  

Key implementations:
- Compared ResNet50 and EfficientNetB0 pretrained on ImageNet
- Implemented feature extraction (frozen backbone + custom classifier head)
- Implemented fine-tuning (unfreezing last 20 layers)
- Applied differential learning rates (1e-3 → 1e-5)
- Built classification head: GAP → Dropout → Dense → Softmax
- Generated confusion matrices and classification reports
- Designed strategy comparison framework

Highlights:
- Small dataset scenario (~200 images per class)
- Demonstrated why transfer learning outperforms training from scratch
- EfficientNetB0 Fine-Tuning achieved best test accuracy: **95.7%**
- Built decision framework for selecting transfer learning strategies
  
---

## 🧠 Activity IX – NLP Sentiment Analysis with Transformers (DistilBERT)

**Dataset:** IMDB Movie Reviews (~5K subset; binary sentiment: positive / negative)  
**Focus:** Natural Language Processing with Transformers, Tokenization, and Fine-Tuning

### Key implementations:

- Loaded and explored the IMDB dataset using the Hugging Face `datasets` library  
- Applied **DistilBERT tokenizer** to perform subword tokenization and generate input tensors  
- Demonstrated **padding, truncation, and attention masks** for transformer inputs  
- Used Hugging Face **pipeline API** for zero-shot sentiment inference  
- Extracted **[CLS] embeddings** from DistilBERT to represent entire reviews  
- Visualized embedding structure using **t-SNE dimensionality reduction**  
- Implemented **attention visualization** to inspect how the model focuses on sentiment-bearing words  
- Built a **PyTorch fine-tuning pipeline** for DistilBERT sequence classification  
- Evaluated model performance using **accuracy, F1 score, confusion matrix, and classification reports**

### Highlights:

- Explored the **Hugging Face ecosystem** (tokenizers, datasets, transformers, pipelines)  
- Demonstrated how **subword tokenization handles unseen vocabulary**  
- Showed that **pretrained transformers capture semantic relationships even before fine-tuning**  
- Visualized how attention weights emphasize sentiment-bearing words (e.g., *superb*, *boring*)  
- Built an end-to-end **transformer fine-tuning workflow using PyTorch and GPU acceleration**

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
