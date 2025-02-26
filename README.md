# 🏥 Diabease - Diabetes Prediction System

![Diabetes](https://img.shields.io/badge/Diabetes-Prediction-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Sklearn-orange?style=for-the-badge&logo=scikitlearn)

## 🌍 Live Demo
🚀 **[Diabease is live!](https://diabease-systems.vercel.app/)**

## 📌 Overview
Diabease is a **machine learning-powered web application** designed to predict diabetes based on user input. The system leverages a trained model to analyze key health parameters and provide accurate predictions.

## ⚡ Features
✅ User-friendly interface for entering health parameters
✅ Machine learning-based prediction model
✅ Hosted on **Vercel** for easy access
✅ Built using **Python, Pandas, NumPy, Sklearn, and Django**

## 📊 Dataset
- **Source:** Kaggle (diabetes.csv)
- **Attributes Used:**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
  - Outcome (0: No Diabetes, 1: Diabetes)

## 🛠 Tech Stack
- **Frontend:** HTML, CSS , JS {Swiper.js)
- **Backend:** Python (Django)
- **Machine Learning Model:** Support Vector Machines (Scikit-learn)
- **Database:** None (uses trained model for inference)
- **Deployment:** Vercel (Frontend), Django (Backend)

## 📌 How to Use
1. Visit **[Diabease](https://diabease-systems.vercel.app/)**
2. Enter the required health details
3. Click **Predict**
4. Get instant diabetes prediction results

## 🖥️ Local Setup
```bash
# Clone the repo
git clone https://github.com/danish-khan962/diabease.git
cd diabease

# Install dependencies
pip install -r requirements.txt

# Run the app
python manage.py runserver
```

## 📈 Visualization - Correlation Heatmap
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("diabetes.csv")
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.show()
```


## 🤝 Contributing
Feel free to **fork** this repository and submit pull requests to enhance the project!

## 📜 License
This project is **open-source** and available under the MIT License.

---

🔗 **Follow Me:** [GitHub](https://github.com/danish-khan962) | [LinkedIn](https://www.linkedin.com/in/danish-khan962/) | [Twitter](#) 🚀

