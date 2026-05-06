import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# 1. Page Config
# ---------------------------
st.set_page_config(page_title="NYPD Crime ML Dashboard", layout="wide")

# ---------------------------
# 2. Load Data
# ---------------------------
@st.cache_data
def get_data():
    df = pd.read_csv("nypd_clean.csv").sample(10000, random_state=42)

    # Extract hour
    df['hour'] = df['cmplnt_fr_tm'].str.split(':').str[0]
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df.loc[df['hour'] == 24, 'hour'] = 0

    return df.dropna(subset=['latitude', 'longitude', 'hour', 'boro_nm', 'ofns_desc'])

df = get_data()

# Encode borough for ML
le_boro = LabelEncoder()
df['boro_code'] = le_boro.fit_transform(df['boro_nm'])

# ---------------------------
# 3. Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs([
    "📍 Hotspot Clustering",
    "🔮 Crime Prediction",
    "⚠️ Anomaly Detection"
])

# =========================================================
# 🔹 TAB 1: HOTSPOT MAP (DBSCAN)
# =========================================================
with tab1:
    st.title("NYPD Crime Hotspot Clustering (DBSCAN)")

    # Sidebar controls
    time_range = st.slider("Select Hour Range:", 0, 23, (12, 20))

    filtered_df = df[
        (df['hour'] >= time_range[0]) &
        (df['hour'] <= time_range[1])
    ].copy()

    if not filtered_df.empty:
        coords = filtered_df[['latitude', 'longitude']].values

        dynamic_min_samples = 15 if len(filtered_df) > 200 else 5

        db = DBSCAN(eps=0.0015, min_samples=dynamic_min_samples).fit(coords)
        filtered_df['cluster'] = db.labels_

        hotspots = filtered_df[filtered_df['cluster'] != -1]

        if hotspots.empty:
            st.info("No dense clusters found. Showing all incidents.")
            map_df = filtered_df
        else:
            st.success(f"{hotspots['cluster'].nunique()} clusters detected")
            map_df = hotspots

        st.map(map_df[['latitude', 'longitude']].rename(
            columns={'latitude': 'lat', 'longitude': 'lon'}
        ))

        st.subheader("Top Offenses")
        st.bar_chart(filtered_df['ofns_desc'].value_counts().head(10))
        
with tab2:
    st.title("Crime Category Prediction (ML)")

    # ---------------------------
    # 1. Create Target Variable FIRST
    # ---------------------------
    def categorize(x):
        if "ASSAULT" in x or "ROBBERY" in x:
            return "Violent"
        elif "LARCENY" in x or "BURGLARY" in x:
            return "Property"
        else:
            return "Public Order"

    df['crime_category'] = df['ofns_desc'].apply(categorize)

    # ---------------------------
    # 2. Features + Labels
    # ---------------------------
    X = df[['hour', 'boro_code']]
    y = df['crime_category']

    # ---------------------------
    # 3. Train/Test Split
    # ---------------------------
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # 4. Train Model
    # ---------------------------
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ---------------------------
    # 5. Model Performance
    # ---------------------------
    st.subheader("Model Performance")

    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Model Accuracy", f"{acc:.2f}")

    with col2:
        st.pyplot(fig, use_container_width=False)

    # ---------------------------
    # 6. Prediction UI
    # ---------------------------
    st.subheader("Predict Crime Type")

    selected_boro = st.selectbox("Select Borough", df['boro_nm'].unique())
    selected_hour = st.slider("Select Hour", 0, 23, 12)

    boro_code = le_boro.transform([selected_boro])[0]

    if st.button("Predict"):
        pred = model.predict([[selected_hour, boro_code]])[0]
        probs = model.predict_proba([[selected_hour, boro_code]])[0]

        st.success(f"Predicted Crime Type: **{pred}**")

        st.write("Prediction Probabilities:")
        st.bar_chart(pd.Series(probs, index=model.classes_))

    # ---------------------------
    # 7. Feature Importance
    # ---------------------------
    st.subheader("Feature Importance (Explainability)")

    importance = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(importance)

# =========================================================
# 🔹 TAB 3: ANOMALY DETECTION (INTERACTIVE)
# =========================================================
with tab3:
    st.title("Crime Anomaly Detection Over Time")

    # --- Filters ---
    st.subheader("Filters")

    selected_boro = st.selectbox(
        "Select Borough",
        ["All"] + list(df['boro_nm'].unique())
    )

    crime_type = st.selectbox(
        "Select Crime Category",
        ["All", "Violent", "Property", "Public Order"]
    )

    # --- Apply filters ---
    filtered = df.copy()

    if selected_boro != "All":
        filtered = filtered[filtered['boro_nm'] == selected_boro]

    # Reuse same categorization logic
    def categorize(x):
        if "ASSAULT" in x or "ROBBERY" in x:
            return "Violent"
        elif "LARCENY" in x or "BURGLARY" in x:
            return "Property"
        else:
            return "Public Order"

    filtered['crime_category'] = filtered['ofns_desc'].apply(categorize)

    if crime_type != "All":
        filtered = filtered[filtered['crime_category'] == crime_type]

    # --- Check if data exists ---
    if filtered.empty:
        st.warning("No data available for selected filters.")
    else:
        # Aggregate crimes per hour
        hourly_counts = (
            filtered.groupby('hour')
            .size()
            .reset_index(name='crime_count')
        )

        # --- Run anomaly detection dynamically ---
        iso = IsolationForest(contamination=0.1, random_state=42)
        hourly_counts['anomaly'] = iso.fit_predict(hourly_counts[['crime_count']])

        anomalies = hourly_counts[hourly_counts['anomaly'] == -1]

        # --- Visualizations ---
        st.subheader("Crime Volume by Hour")
        st.line_chart(hourly_counts.set_index('hour')['crime_count'])

        st.subheader("Detected Anomalies")
        st.dataframe(anomalies)

        # Highlight anomaly hours clearly
        if not anomalies.empty:
            st.info(
                f"Detected {len(anomalies)} anomalous time periods "
                f"(unusual spikes or drops in crime)."
            )
        else:
            st.success("No significant anomalies detected for this selection.")