import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

df = pd.read_csv("data/crowdfunding_cleaned.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SELECT PRE-LAUNCH FEATURES
# ==========================================

features = [
    "Goal_Amount",
    "Category",
    "Subcategory",
    "Country",
    "Duration_Days",
    "Title_Length",
    "Launch_Month",
    "Launch_Year",
    "Launch_Day_of_Week"
]

target = "Success"

X = df[features]
y = df[target]


print("\nFeatures used:")
print(features)

print("\nTarget:")
print(target)


# ==========================================
# 3. IDENTIFY FEATURE TYPES
# ==========================================

numeric_features = [
    "Goal_Amount",
    "Duration_Days",
    "Title_Length",
    "Launch_Month",
    "Launch_Year"
]

categorical_features = [
    "Category",
    "Subcategory",
    "Country",
    "Launch_Day_of_Week"
]


# ==========================================
# 4. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=5
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ==========================================
# 5. CREATE LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


# ==========================================
# 6. CREATE COMPLETE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nTraining Logistic Regression model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 9. PREDICTIONS
# ==========================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]


# ==========================================
# 10. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n==========================================")
print("       MODEL PERFORMANCE")
print("==========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n==========================================")
print("       CONFUSION MATRIX")
print("==========================================")

print(cm)


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("       CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Failed",
            "Successful"
        ],
        zero_division=0
    )
)


# ==========================================
# 13. TEST SAMPLE CAMPAIGN
# ==========================================

sample_campaign = pd.DataFrame({
    "Goal_Amount": [5000],
    "Category": ["Technology"],
    "Subcategory": ["Apps"],
    "Country": ["US"],
    "Duration_Days": [30],
    "Title_Length": [35],
    "Launch_Month": [9],
    "Launch_Year": [2017],
    "Launch_Day_of_Week": ["Monday"]
})


sample_prediction = pipeline.predict(
    sample_campaign
)[0]

sample_probability = pipeline.predict_proba(
    sample_campaign
)[0][1]


print("\n==========================================")
print("       SAMPLE CAMPAIGN PREDICTION")
print("==========================================")

print(
    "Success Probability:",
    f"{sample_probability * 100:.2f}%"
)

if sample_prediction == 1:
    print("Prediction: LIKELY SUCCESSFUL")
else:
    print("Prediction: LIKELY UNSUCCESSFUL")


# ==========================================
# 14. SAVE MODEL
# ==========================================

joblib.dump(
    pipeline,
    "crowdfunding_model.pkl"
)

print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY!")
print("File: crowdfunding_model.pkl")
print("==========================================")