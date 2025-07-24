# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
# STEP 2: LOAD DATA

df = pd.read_csv("customerChurn.csv")
print("Data loaded successfully.\n")
print(df.head())
# STEP 3: DATA CLEANING

# Convert 'TotalCharges' to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing 'TotalCharges' with the median value (safe way)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Convert 'Churn' column to binary: Yes → 1, No → 0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop 'customerID' column ONLY if it exists
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Optional: Print confirmation
print(" Data cleaning complete.")
print("Remaining columns:\n", df.columns)
print("Preview of cleaned data:\n", df.head())
# STEP 4: ENCODING CATEGORICAL COLUMNS

# Identify categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()

# Label encode binary columns
binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode the remaining
df = pd.get_dummies(df, columns=[col for col in cat_cols if col not in binary_cols])

print(df.head())
# STEP 5: FEATURE SCALING

scaler = StandardScaler()
df[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['MonthlyCharges', 'TotalCharges']])
print(df[['MonthlyCharges', 'TotalCharges']].head())
# STEP 6: TRAIN-TEST SPLIT

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# STEP 7: TRAINING LOGISTIC MODEL

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# STEP 8: MODEL EVALUATION

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# STEP 9: SAVE THE MODEL

joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\n Model saved as logistic_model.pkl")