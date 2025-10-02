Logistic Regression from Scratch (NumPy)

This project implements Logistic Regression from scratch using only NumPy. The model is trained and tested on the Breast Cancer dataset from scikit-learn and compared against scikit-learn’s built-in implementation.

Features: 
- Sigmoid function with numerical stability (np.clip)
- Gradient descent optimization
- Methods for training (fit), prediction (predict), and evaluation (score)
- Benchmark against scikit-learn’s LogisticRegression

## 📂 Project Files

- **`logistic_regression.py`** → Contains the Logistic Regression class implemented from scratch using NumPy. Includes methods for `fit`, `predict`, `score`, and `predict_proba`.  
- **`demo_breast_cancer.ipynb`** → Jupyter Notebook demo. Loads the Breast Cancer dataset, trains the custom model, compares accuracy with scikit-learn’s LogisticRegression, and plots the learning curve.  
- **`README.md`** → Overview of the project, features, results, and instructions to run.  
