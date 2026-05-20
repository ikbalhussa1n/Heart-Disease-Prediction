# ❤️ Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🧠 Overview

This project predicts the likelihood of heart disease using a Machine Learning model (K-Nearest Neighbors).  
It is deployed using a Streamlit web application for easy and interactive predictions.

---

## 🚀 Live Demo
(Add your Streamlit link here)

---

## 📊 Dataset Features

- age: Age of patient  
- sex: Gender (1 = male, 0 = female)  
- cp: Chest pain type  
- thalach: Maximum heart rate achieved  
- exang: Exercise induced angina  
- oldpeak: ST depression  
- slope: Slope of peak exercise ST segment  
- ca: Number of major vessels  
- thal: Thalassemia  
- target: 0 = No Disease, 1 = Disease  

---

## 🛠️ Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- StandardScaler  
- KNeighborsClassifier  
- Pickle  
- Streamlit  

---

## 🧪 Machine Learning Pipeline

- Data preprocessing  
- Feature selection  
- Train-test split  
- Feature scaling using StandardScaler  
- Model training using KNN  
- Hyperparameter tuning (k = 1–15)  
- Model saved using pickle  

---

## 📸 App Preview

<img width="960" height="423" alt="heartPred" src="https://github.com/user-attachments/assets/25e1a689-6ef7-42c0-92c6-a204e75be2d3" />


---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/ikbalhussa1n/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

2. Create virtual environment
python -m venv .venv
3. Activate environment

Windows
.venv\Scripts\activate

Mac/Linux
source .venv/bin/activate

4. Install dependencies
pip install -r requirements.txt

5. Run Streamlit app
streamlit run app.py

📁 Project Structure
Heart-Disease-Prediction/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── dataset.csv
├── requirements.txt
└── README.md

📈 Results
Model Accuracy: ~88%
Best k selected using validation curve
StandardScaler improved performance significantly


👨‍💻 Author
Md Ikbal Hussain
