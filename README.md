# 🚀 Customer Churn ML Model Comparison

## 📌 Project Overview

This project implements and compares three machine learning models for customer churn prediction and classification tasks.

It was developed as part of an Advanced Machine Learning Internship (Level 3).

The project demonstrates:

✔ Data preprocessing & feature engineering  
✔ Model training & hyperparameter tuning  
✔ Cross-validation  
✔ Performance comparison  
✔ Visualisation of results  
✔ Neural network training with TensorFlow  

---

## 🧠 Models Implemented

### 🔹 1. Random Forest Classifier
- GridSearchCV hyperparameter tuning
- 5-fold cross-validation
- Feature importance analysis
- Confusion matrix visualization
- Precision, Recall, F1-score, Accuracy

---

### 🔹 2. Support Vector Machine (SVM)
- Linear Kernel
- RBF Kernel
- Decision boundary visualization
- ROC curve comparison
- AUC score evaluation

---

### 🔹 3. Neural Network (TensorFlow / Keras)
- Fully connected Dense layers
- Dropout regularization
- Early stopping
- Binary cross-entropy loss
- Adam optimizer
- Training vs Validation curves

(Fallback to sklearn MLPClassifier if TensorFlow is not available)

---

## 📊 Output Visualizations

The following plots are automatically generated:

- `task1_random_forest_results.png`
- `task2_svm_results.png`
- `task2_svm_roc.png`
- `task3_neural_network_results.png`

These include:
- Confusion matrices
- Feature importance
- Decision boundaries
- ROC curves
- Training/Validation loss & accuracy curves

---

## 📁 Project Structure
Customer-Churn-ML-Model-Comparison/
│
├── level3_all_tasks.py
├── churn-bigml-80.csv
├── churn-bigml-20.csv
├── iris.csv
├── requirements.txt
├── task1_random_forest_results.png
├── task2_svm_results.png
├── task2_svm_roc.png
└── task3_neural_network_results.png


---

## ▶️ How To Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Rohit2426-bit/Customer-Churn-ML-Model-Comparison.git

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run Project

python level3_all_tasks.py


---

# 🔥 OPTIONAL: Add Badges (Looks More Professional)

Add this at the very top of README:

```markdown
![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-red)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)
