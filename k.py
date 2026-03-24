# =========================================================
# AUTO ML STUDIO — USER REQUESTED HERO UI STYLE
# LOGIC: EXACTLY SAME
# UI: DARK GRADIENT + HERO + CARDS (AS REQUESTED)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import time

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Unsupervised
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# USER PROVIDED UI STYLE
# =========================================================

st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
}

.hero {
background: linear-gradient(
135deg,
rgba(102,126,234,0.6),
rgba(118,75,162,0.6)
);

padding:40px;
border-radius:20px;
box-shadow:0 10px 40px rgba(0,0,0,0.4);
}

.card {
background: rgba(255,255,255,0.05);
padding:20px;
border-radius:15px;
text-align:center;
font-size:18px;
font-weight:600;
}

.upload-box {
background: linear-gradient(
135deg,
#f6d365,
#fda085
);

padding:20px;
border-radius:15px;
text-align:center;
font-weight:600;
}

.section {
background: rgba(255,255,255,0.04);
padding:20px;
border-radius:18px;
margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">
<h1>🚀 AutoML Studio</h1>
<h3>🔥 Build Machine Learning Models in Seconds</h3>
<p>
Upload datasets, preprocess, train & predict — all in one place.
</p>
<p>
👉 No coding • No complexity • Just powerful AI
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# FEATURE CARDS
# =========================================================

c1, c2, c3 = st.columns(3)

c1.markdown(
    '<div class="card">⚡ Instant Training</div>',
    unsafe_allow_html=True
)

c2.markdown(
    '<div class="card">🤖 15+ Models</div>',
    unsafe_allow_html=True
)

c3.markdown(
    '<div class="card">📊 Optimized Results</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 📂 Upload Dataset")

file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# =========================================================
# MAIN LOGIC (UNCHANGED FROM ORIGINAL)
# =========================================================

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("✅ Dataset Loaded Successfully")

    # ================= DATA OVERVIEW =================

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    m1.metric("Rows", df.shape[0])
    m2.metric("Columns", df.shape[1])
    m3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_cols) > 0:

        st.markdown(
            '<div class="section">',
            unsafe_allow_html=True
        )

        col = st.selectbox(
            "📈 Distribution Column",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=col
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ================= PREPROCESSING =================

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True
    )

    fill_cols = st.multiselect(
        "Columns",
        df.columns
    )

    fill_method = st.selectbox(
        "Method",
        [
            "Mean",
            "Median",
            "Mode",
            "Forward Fill",
            "Backward Fill"
        ]
    )

    if st.button("Apply Missing Fill"):

        for col in fill_cols:

            if fill_method == "Mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())

            elif fill_method == "Median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())

            elif fill_method == "Mode":
                df[col] = df[col].fillna(df[col].mode()[0])

            elif fill_method == "Forward Fill":
                df[col] = df[col].ffill()

            elif fill_method == "Backward Fill":
                df[col] = df[col].bfill()

        st.session_state.df = df

        st.success(
            "✅ Missing Values Handled"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    learning_type = st.radio(
        "🧠 Select Learning Type",
        [
            "Supervised",
            "Unsupervised"
        ]
    )

    # ================= SUPERVISED =================

    if learning_type == "Supervised":

        target = st.selectbox(
            "Target Column",
            df.columns
        )

        df = df.dropna(
            subset=[target]
        )

        X = df.drop(
            columns=[target]
        )

        y = df[target]

        target_type = type_of_target(y)

        if target_type in [
            "binary",
            "multiclass"
        ]:
            task = "Classification"
        else:
            task = "Regression"

        st.info(
            f"🎯 Task: {task}"
        )

        k = st.slider(
            "Top K Features",
            1,
            X.shape[1],
            min(5, X.shape[1])
        )

        selector = SelectKBest(
            f_classif if task == "Classification" else f_regression,
            k=k
        )

        X_new = selector.fit_transform(
            X,
            y
        )

        selected_features = X.columns[
            selector.get_support()
        ]

        X = pd.DataFrame(
            X_new,
            columns=selected_features
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        results = []

        progress = st.progress(0)

        best_model = None
        best_model_name = None

        best_score = 0

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True),
            "Naive Bayes": GaussianNB()
        }

        for i, (name, model) in enumerate(models.items()):

            model.fit(
                X_train,
                y_train
            )

            preds = model.predict(
                X_test
            )

            acc = accuracy_score(
                y_test,
                preds
            )

            results.append([
                name,
                acc
            ])

            progress.progress(
                (i + 1) / len(models)
            )

            if acc > best_score:

                best_score = acc
                best_model = model
                best_model_name = name

        res = pd.DataFrame(
            results,
            columns=[
                "Model",
                "Accuracy"
            ]
        )

        st.dataframe(
            res,
            use_container_width=True
        )

        st.success(
            f"Best Model Selected: {best_model_name}"
        )

    # ================= UNSUPERVISED =================

    else:

        data = df.select_dtypes(include=np.number)

        scaler = StandardScaler()

        data_scaled = scaler.fit_transform(data)

        models = {
            "KMeans": KMeans(n_clusters=3),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
            "Birch": Birch(n_clusters=3),
            "DBSCAN": DBSCAN()
        }

        results = []
        best_score = -1

        progress = st.progress(0)

        for i, (name, model) in enumerate(models.items()):

            labels = model.fit_predict(data_scaled)

            if len(set(labels)) > 1:

                score = silhouette_score(
                    data_scaled,
                    labels
                )

            else:

                score = -1

            results.append([name, score])

            progress.progress(
                (i + 1) / len(models)
            )

            if score > best_score:

                best_score = score
                best_model_name = name
                best_labels = labels

        res = pd.DataFrame(
            results,
            columns=[
                "Algorithm",
                "Silhouette Score"
            ]
        )

        st.dataframe(
            res,
            use_container_width=True
        )

        st.success(
            f"Best Clustering Model: {best_model_name}"
        )

else:

    # ---------------- FRONT PAGE (USER HERO UI ONLY WHEN NO DATASET) ----------------

    st.markdown("""
    <div class="hero">
    <h1>🚀 AutoML Studio</h1>
    <h3>🔥 Build Machine Learning Models in Seconds</h3>
    <p>Upload datasets, preprocess, train & predict — all in one place.</p>
    <p>👉 No coding • No complexity • Just powerful AI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)

    c1.markdown(
        '<div class="card">⚡ Instant</div>',
        unsafe_allow_html=True
    )

    c2.markdown(
        '<div class="card">🤖 8+ Models</div>',
        unsafe_allow_html=True
    )

    c3.markdown(
        '<div class="card">📊 Optimized</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.info("👈 Upload your dataset from the sidebar to start AutoML.")
