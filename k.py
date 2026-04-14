import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

from sklearn.metrics import accuracy_score, mean_squared_error

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# PREMIUM UI STYLE
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg,#020617,#020617);
}

/* HERO */

.hero {
    background: linear-gradient(135deg,#2563eb,#7c3aed,#9333ea);
    padding: 42px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 18px 40px rgba(124,58,237,0.35);
}

.hero-title {
    font-size: 46px;
    font-weight: 900;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: #e2e8f0;
    margin-top: 10px;
}

/* FEATURE BOX */

.feature-box {
    background: rgba(255,255,255,0.05);
    padding: 16px;
    border-radius: 14px;
    text-align: center;
    font-weight: 600;
    color: white;
    transition: 0.25s;
}

.feature-box:hover {
    transform: translateY(-6px);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#0f172a);
}

/* BUTTON */

.stButton>button {
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    color: white;
    border-radius: 10px;
    font-weight: 600;
}

/* FOOTER */

.footer {
    margin-top: 50px;
    padding: 16px;
    text-align: center;
    color: #9ca3af;
    font-size: 14px;
    border-top: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO
# =====================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🚀 AutoML Studio
</div>

<div class="hero-subtitle">
Enterprise-Grade Machine Learning Automation • Intelligent Model Optimization • Advanced Predictive Intelligence Platform
</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# FEATURE BOXES
# =====================================================

f1, f2, f3, f4 = st.columns(4)

f1.markdown('<div class="feature-box">⚡ High-Performance Training Engine</div>', unsafe_allow_html=True)
f2.markdown('<div class="feature-box">🤖 Intelligent Model Optimization</div>', unsafe_allow_html=True)
f3.markdown('<div class="feature-box">📊 Advanced Data Intelligence</div>', unsafe_allow_html=True)
f4.markdown('<div class="feature-box">☁️ Production-Ready Deployment</div>', unsafe_allow_html=True)

# =====================================================
# FILE UPLOAD
# =====================================================

st.sidebar.markdown("## 📂 Upload Dataset")

file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if file:

    df = pd.read_csv(file)

    st.success("Dataset Loaded Successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------- Preprocessing ----------------

    st.subheader("Preprocessing")

    fill_cols = st.multiselect(
        "Columns",
        df.columns
    )

    fill_method = st.selectbox(
        "Method",
        ["Mean","Median","Mode"]
    )

    if st.button("Apply Missing Fill"):

        for col in fill_cols:

            if fill_method == "Mean":
                df[col] = df[col].fillna(df[col].mean())

            elif fill_method == "Median":
                df[col] = df[col].fillna(df[col].median())

            elif fill_method == "Mode":
                df[col] = df[col].fillna(df[col].mode()[0])

        st.success("Missing Values Handled")

    # ---------------- Encoding ----------------

    cat_cols = df.select_dtypes(
        include="object"
    ).columns

    encode_cols = st.multiselect(
        "Categorical Columns",
        cat_cols
    )

    if st.button("Apply Encoding"):

        for col in encode_cols:

            df[col] = LabelEncoder().fit_transform(
                df[col]
            )

        st.success("Encoding Applied")

    # ---------------- Scaling ----------------

    num_cols = df.select_dtypes(
        include=np.number
    ).columns

    scale_cols = st.multiselect(
        "Columns for Scaling",
        num_cols
    )

    if st.button("Apply Scaling"):

        scaler = StandardScaler()

        df[scale_cols] = scaler.fit_transform(
            df[scale_cols]
        )

        st.success("Scaling Applied")

    # =====================================================
    # LEARNING TYPE
    # =====================================================

    learning_type = st.radio(
        "Select Learning Type",
        ["Supervised","Unsupervised"]
    )

    # =====================================================
    # SUPERVISED
    # =====================================================

    if learning_type == "Supervised":

        target = st.selectbox(
            "Target Column",
            df.columns
        )

        X = df.drop(columns=[target])
        y = df[target]

        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=0.2
        )

        st.subheader("Model Leaderboard")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB()
        }

        results = []
        best_score = 0

        with st.spinner("Training models..."):

            for name, model in models.items():

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                acc = accuracy_score(
                    y_test,
                    preds
                )

                results.append([name, acc])

                if acc > best_score:

                    best_score = acc
                    best_model = model
                    best_model_name = name

        leaderboard = pd.DataFrame(
            results,
            columns=["Model","Accuracy"]
        )

        st.dataframe(leaderboard)

        st.success(
            f"Best Model: {best_model_name}"
        )

    # =====================================================
    # UNSUPERVISED
    # =====================================================

    else:

        st.subheader("Clustering Leaderboard")

        data = df.select_dtypes(
            include=np.number
        )

        scaler = StandardScaler()

        data_scaled = scaler.fit_transform(
            data
        )

        models = {
            "KMeans": KMeans(n_clusters=3),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
            "Birch": Birch(n_clusters=3),
            "DBSCAN": DBSCAN()
        }

        results = []

        for name, model in models.items():

            labels = model.fit_predict(
                data_scaled
            )

            if len(set(labels)) > 1:

                score = silhouette_score(
                    data_scaled,
                    labels
                )

            else:

                score = -1

            results.append([name, score])

        res = pd.DataFrame(
            results,
            columns=[
                "Algorithm",
                "Silhouette Score"
            ]
        )

        st.dataframe(res)

else:

    st.info(
        "Upload dataset to start AutoML"
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown(
    '<div class="footer">Developed by Krishna Gediya | AutoML Application</div>',
    unsafe_allow_html=True
)
