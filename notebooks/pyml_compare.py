import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / "nypd_sampled.csv")

df["cmplnt_fr_dt"] = pd.to_datetime(df["cmplnt_fr_dt"])
df["cmplnt_fr_tm"] = pd.to_datetime(df["cmplnt_fr_tm"], format="%H:%M:%S")

df["high_risk_label"] = (df["law_cat_cd"].str.upper() == "FELONY").astype(int)

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

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

results = []
trained_models = {}

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba)
    })

    trained_models[name] = pipe

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
results_df.to_csv(base_dir / "model_comparison.csv", index=False)

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

best_pred = best_model.predict(X_test)
best_proba = best_model.predict_proba(X_test)[:, 1]

master_ml_results = model_df.loc[idx_test, export_cols].copy().reset_index(drop=True)
master_ml_results["actual_label"] = y_test.reset_index(drop=True)
master_ml_results["predicted_label"] = best_pred
master_ml_results["predicted_probability"] = best_proba
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

master_ml_results["final_model"] = best_model_name

master_ml_results.to_csv(base_dir / "master_ml_results.csv", index=False)

print(results_df)
print("\nBest model:", best_model_name)
print("saved: model_comparison.csv")
print("saved: master_ml_results.csv")

from sklearn.metrics import roc_curve

roc_rows = []

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba)
    })

    fpr, tpr, thresholds = roc_curve(y_test, proba)

    for i in range(len(fpr)):
        roc_rows.append({
            "model": name,
            "fpr": fpr[i],
            "tpr": tpr[i],
            "threshold": thresholds[i]
        })

    trained_models[name] = pipe

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
results_df.to_csv(base_dir / "model_comparison.csv", index=False)

roc_curve_df = pd.DataFrame(roc_rows)
roc_curve_df.to_csv(base_dir / "roc_curve_points.csv", index=False)