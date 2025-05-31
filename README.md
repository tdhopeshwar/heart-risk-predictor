# Heart Risk Predictor 🫀
Predicting 10-Year Coronary Heart Disease (CHD) Risk using Logistic Regression from Scratch

## 📌 Overview
This project implements a custom logistic regression model to predict the risk of developing coronary heart disease (CHD) within 10 years. The pipeline includes data cleaning, imputation, normalization, class imbalance handling, and performance evaluation—built entirely from scratch without using pre-built ML libraries like `scikit-learn`.

## 🧪 Dataset
The dataset is based on the Framingham Heart Study and includes clinical features such as:
- Age, gender, education
- Systolic/diastolic blood pressure
- BMI, cholesterol, glucose, smoking habits
- Presence of diabetes and prevalent hypertension
- Target: `TenYearCHD` (binary: 0 = no risk, 1 = at risk)

## 🔍 Problem
The dataset is highly imbalanced, with most patients not experiencing CHD. Therefore, the goal is to build a model that emphasizes **recall** (sensitivity), minimizing false negatives to ensure high-risk individuals are not overlooked.

## 🛠️ Pipeline
### 1. Exploratory Data Analysis (EDA)
- Plotted distributions of BP medications, cholesterol, glucose, BMI, and heart rate across genders and conditions.
- Assessed missing value patterns.

### 2. Missing Value Imputation
- Used a **subclass-based method** with domain-relevant grouping:
  - Example: `BPMeds` imputed by `prevalentHyp`, `totChol` by `gender`
- Supported strategies: `mean`, `mode`, `median`

### 3. Feature Normalization
- Applied Z-Score normalization to numeric columns to standardize feature scales.

### 4. Class Imbalance Handling
- Performed oversampling of minority class (CHD = 1) to create a balanced dataset.

### 5. Logistic Regression from Scratch
- Implemented `sigmoid`, `cost`, and `gradient descent` manually using NumPy
- Trained on balanced data
- Set classification threshold to `0.4` to increase recall

### 6. Evaluation
- Metrics: Accuracy, Precision, Recall
- Plotted confusion matrices for both train and test sets

## 📊 Results
| Metric     | Train Set | Test Set |
|------------|-----------|----------|
| Accuracy   | 0.59      | 0.37     |
| Precision  | 0.55      | 0.19     |
| Recall     | **0.93**  | **0.96** |

> **Interpretation**: In a medical context, high recall is prioritized to reduce the chance of missing CHD cases. The model trades off precision in favor of sensitivity, which is an acceptable and necessary compromise for healthcare screening models.

## 📁 Files
- `CHD_Prediction_Logistic_Regression.ipynb`: Full code with explanations, visualizations, and evaluations.
- `README.md`: Project summary and structure.
- `subclass_imputation.py` *(optional)*: Contains reusable imputation function.

## 💡 Key Takeaways
- Built a full ML pipeline from scratch without relying on external ML libraries.
- Used clinically appropriate data processing techniques.
- Successfully balanced a sensitive model for real-world healthcare relevance.

## 📬 Contact
For any questions or collaborations, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/tanishadhopeshwar).

---

