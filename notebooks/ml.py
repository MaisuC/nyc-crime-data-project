import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / "nypd_clean.csv")

df["cmplnt_fr_dt"] = pd.to_datetime(df["cmplnt_fr_dt"])
df["cmplnt_fr_tm"] = pd.to_datetime(df["cmplnt_fr_tm"], format="%H:%M:%S")

# target: felony = high risk
df["high_risk_label"] = (df["law_cat_cd"].str.upper() == "FELONY").astype(int)

# simple time features
df["complaint_hour"] = df["cmplnt_fr_tm"].dt.hour
df["complaint_dayofweek"] = df["cmplnt_fr_dt"].dt.day_name()
df["complaint_month"] = df["cmplnt_fr_dt"].dt.month
df["complaint_year"] = df["cmplnt_fr_dt"].dt.year
df["year_month"] = df["cmplnt_fr_dt"].dt.to_period("M").astype(str)

feature_cols = [
    "boro_nm",
    "patrol_boro",
    "susp_age_group",
    "susp_race",
    "susp_sex",
    "vic_age_group",
    "vic_race",
    "vic_sex",
    "crm_atpt_cptd_cd",
    "complaint_hour",
    "complaint_dayofweek",
    "complaint_month",
    "complaint_year",
]

export_cols = [
    "cmplnt_fr_dt",
    "year_month",
    "boro_nm",
    "ofns_desc",
    "patrol_boro",
    "susp_age_group",
    "susp_race",
    "susp_sex",
    "vic_age_group",
    "vic_race",
    "vic_sex",
    "latitude",
    "longitude",
    "high_risk_label",
]

model_df = df[feature_cols + export_cols].copy()
model_df = model_df.loc[:, ~model_df.columns.duplicated()]

X = model_df[feature_cols]
y = model_df["high_risk_label"]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, model_df.index, test_size=0.2, random_state=42, stratify=y
)

categorical_features = [
    "boro_nm",
    "patrol_boro",
    "susp_age_group",
    "susp_race",
    "susp_sex",
    "vic_age_group",
    "vic_race",
    "vic_sex",
    "crm_atpt_cptd_cd",
    "complaint_dayofweek",
]

numeric_features = [
    "complaint_hour",
    "complaint_month",
    "complaint_year",
]

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=300))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
auc = roc_auc_score(y_test, proba)
cm = confusion_matrix(y_test, pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("ROC AUC:", auc)
print("Confusion Matrix:")
print(cm)

master_ml_results = model_df.loc[idx_test, export_cols].copy().reset_index(drop=True)
master_ml_results["actual_label"] = y_test.reset_index(drop=True)
master_ml_results["predicted_label"] = pred
master_ml_results["predicted_probability"] = proba
master_ml_results["prediction_correct"] = (
    master_ml_results["actual_label"] == master_ml_results["predicted_label"]
).astype(int)

master_ml_results["error_type"] = "FN"
master_ml_results.loc[
    (master_ml_results["actual_label"] == 1) & (master_ml_results["predicted_label"] == 1),
    "error_type"
] = "TP"
master_ml_results.loc[
    (master_ml_results["actual_label"] == 0) & (master_ml_results["predicted_label"] == 0),
    "error_type"
] = "TN"
master_ml_results.loc[
    (master_ml_results["actual_label"] == 0) & (master_ml_results["predicted_label"] == 1),
    "error_type"
] = "FP"

master_ml_results.to_csv(base_dir / "master_ml_results.csv", index=False)

print("saved: master_ml_results.csv")
print(master_ml_results.head())