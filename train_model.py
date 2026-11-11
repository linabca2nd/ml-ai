# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. LOAD DATASET

df = pd.read_csv("data/diabetes_prediction_dataset.csv")

print("\nFIRST FIVE RECORDS")
print(df.head())

# 2. UNDERSTAND DATA

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL DESCRIPTION")
print(df.describe())

# 3. CHECK MISSING VALUES

print("\nMISSING VALUES")
print(df.isnull().sum())

# 4. CHECK DUPLICATES

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# 5. REMOVE DUPLICATES

df = df.drop_duplicates()

print("\nDATA AFTER CLEANING")
print(df.shape)

# 6. CHECK TARGET DISTRIBUTION

print("\nDIABETES DISTRIBUTION")
print(df["diabetes"].value_counts())

# 7. EXPLORATORY DATA ANALYSIS

# Diabetes distribution graph

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="diabetes")
plt.title("Diabetes Distribution")
plt.savefig("diabetes_distribution.png")
plt.show()

# Age distribution graph

plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="age", hue="diabetes", bins=30)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.show()

# Correlation heatmap

plt.figure(figsize=(9, 6))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# 8. SEPARATE INPUT AND OUTPUT

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# 9. IDENTIFY COLUMNS

numeric_features = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

categorical_features = [
    "gender",
    "smoking_history"
]

# 10. PREPROCESSING

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 11. SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# 12. MACHINE LEARNING MODELS

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}

# 13. TRAIN AND EVALUATE MODELS

results = []
trained_models = {}

for name, classifier in models.items():

    print("\nTRAINING:", name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    trained_models[name] = pipeline

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

# 14. COMPARE MODELS

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

# 15. MODEL COMPARISON GRAPH

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(10, 6))

plt.title("Machine Learning Model Comparison")
plt.ylabel("Performance Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# 16. SELECT BEST MODEL

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBEST MODEL:", best_model_name)

# 17. SAVE MODEL

joblib.dump(best_model, "diabetes_model.pkl")

print("\nMODEL SAVED SUCCESSFULLY!")

# 18. CONFUSION MATRIX

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 19. CLASSIFICATION REPORT

print("\nCLASSIFICATION REPORT")

print(classification_report(y_test, best_predictions))

print("\nPROJECT TRAINING COMPLETED!")