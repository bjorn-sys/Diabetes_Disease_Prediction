--------------------------------------------------------------

🩺 Diabetes Risk Prediction Using Machine Learning

--------------------------------------------------------------

📘 Project Overview

This project develops a machine learning model to predict the likelihood of diabetes based on key health and lifestyle indicators such as glucose levels, BMI, blood pressure, insulin levels, and age.
The primary objective is to optimize for recall — ensuring that potential diabetes cases are rarely missed — while maintaining high overall accuracy and interpretability.

--------------------------------------------------------------

📂 Dataset Description

The dataset used is the Pima Indians Diabetes Database, containing 768 patient records and the following attributes:

Feature	Description
Pregnancies	Number of pregnancies
Glucose	Plasma glucose concentration
BloodPressure	Diastolic blood pressure (mm Hg)
SkinThickness	Triceps skin fold thickness (mm)
Insulin	2-hour serum insulin (mu U/ml)
BMI	Body Mass Index (kg/m²)
DiabetesPedigreeFunction	Diabetes heredity measure
Age	Age of patient in years
Outcome	Target variable (1 = Diabetes, 0 = No Diabetes)

--------------------------------------------------------------

⚙️ Data Preprocessing Steps
1️⃣ Data Loading and Exploration

Loaded and explored the dataset

Checked datatypes, null values, and target distribution

Created a correlation heatmap for feature relationships

2️⃣ Data Splitting

Divided data into training and testing sets

3️⃣ Data Scaling

Standardized all features for better model performance

4️⃣ Handling Class Imbalance

Applied SMOTE to balance diabetic vs. non-diabetic cases

--------------------------------------------------------------

🧠 Models Implemented
Model	Description	Mean Accuracy	Test Accuracy
Logistic Regression	Linear baseline classifier	0.76	0.77
K-Nearest Neighbors (KNN)	Distance-based classifier	0.73	0.75 (Best)
Random Forest	Ensemble of decision trees	0.75	0.76
Decision Tree	Single non-linear classifier	0.69	0.70
XGBoost	Gradient boosting ensemble	0.73	0.74
CatBoost	Boosting optimized for categorical data	0.72	0.75
Support Vector Machine (SVM)	Kernel-based classifier	0.72	0.74

--------------------------------------------------------------

🧩 Model Optimization
🎯 Best Performing Model

K-Nearest Neighbors (KNN) achieved the highest test accuracy and stability across folds.

⚙️ Further Recall Optimization

Used Logistic Regression and XGBoost with:

Hyperparameter tuning

Class balancing

Threshold tuning to increase recall and reduce false negatives

--------------------------------------------------------------

📊 Evaluation Metrics
Metric	Description	Result
Accuracy	Overall correct predictions	0.77
Recall	True positive rate (critical for diagnosis)	0.74
Precision	Accuracy of positive predictions	0.65
F1-Score	Harmonic mean of precision & recall	0.68
ROC-AUC	Model discrimination capability	0.75

Classification Report Example (Threshold = 0.4):

Precision for non-diabetic: 0.91

Recall for non-diabetic: 0.58

Precision for diabetic: 0.53

Recall for diabetic: 0.89

Accuracy: 0.69

--------------------------------------------------------------

📈 Visualizations Included

🧮 Correlation Heatmap — reveals feature dependencies

📉 Confusion Matrix — visualizes prediction results

🧩 ROC Curve — evaluates classifier discrimination

--------------------------------------------------------------

💾 Model Export and Deployment

Trained and optimized models are serialized for deployment

Can be integrated into Streamlit or other systems for live predictions

--------------------------------------------------------------

🧠 Key Learnings

✅ Standardization and SMOTE significantly improved model balance
✅ Lowering classification threshold improved recall for medical accuracy
✅ KNN and XGBoost provided consistent, interpretable results across folds

---------------

LINK: [https://diabetesdiseaseprediction.streamlit.app/
