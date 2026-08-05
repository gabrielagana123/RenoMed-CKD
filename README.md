# RenoMed-CKD Predictor

A machine learning project that predicts the likelihood of Chronic Kidney Disease (CKD) using patient clinical data.
# Live Demo
## Project Overview

Chronic Kidney Disease is a major global health concern that can lead to kidney failure if not detected early. This project uses machine learning techniques to predict CKD risk based on clinical laboratory measurements and patient health indicators.

The goal of this project is to build a Machine Learning Model that predicts the likelihood of Chronic Kidney Disease
---

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Multiple model comparison
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Model evaluation using healthcare-focused metrics
- Feature importance analysis
- SHAP explainability
- CKD risk categorization
- Interactive Streamlit web application

---

## Dataset

Dataset Source:
- Kaggle CKD Dataset: https://www.kaggle.com/datasets/mansoordaku/ckdisease

Description: The data was taken over a 2-month period in India with 25 features ( eg, red blood cell count, white blood cell count, etc). The target is the 'classification', which is either 'ckd' or 'notckd' - ckd=chronic kidney disease. There are 400 rows
The data needs cleaning: in that it has NaNs and the numeric features need to be forced to floats. According to the source, they were instructed to get rid of ALL ROWS with Nans, with no threshold. Meaning, any row that has even one NaN, gets deleted.

Features include:
- Age
- Blood Pressure
- Albumin
- Sugar
- Blood Glucose Random
- Serum Creatinine
- Sodium
- Potassium
- Hemoglobin
- White Blood Cell Count

Target:
- CKD
- NotCKD

---

## Machine Learning Workflow

1. Data Cleaning
2. Missing Value Handling
3. Feature Encoding
4. Exploratory Data Analysis
5. Model Training
6. Model Comparison
7. Evaluation
8. Explainability Analysis
9. Deployment

---

## Evaluation Metrics

This project evaluates models using:



In healthcare systems, recall is especially important because missing a CKD-positive patient could delay treatment and increase health risks.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP
- Streamlit

---

## Results

Random Forest achieved the best performance among the tested models and demonstrated strong predictive capability for CKD classification.

Key predictive features included:


---

## Future Improvements



---

## Author

Gabriel Agana

Interested in healthcare AI, machine learning, and predictive analytics.
