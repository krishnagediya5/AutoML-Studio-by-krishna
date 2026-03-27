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

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="AutoML Studio Enterprise",
    page_icon="🏢",
    layout="wide"
)

# ----------------------------------------------------
# ENTERPRISE UI THEME
# ----------------------------------------------------

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.block-container {
    padding-top: 1rem;
}

/* Header */

.enterprise-header {
    background: linear-gradient(90deg,#0ea5e9,#6366f1);
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.header-title {
    font-size: 32px;
    font-weight: bold;
    color: white;
}

.header-subtitle {
    font-size: 15px;
    color: #e2e8f0;
}

/* Cards */

.metric-card {
    background: #1e293b;
    padding: 18px;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}

/* Buttons */

.stButton>button {
    background: linear-gradient(90deg,#0ea5e9,#22c55e);
    color: white;
    border-radius: 8px;
    height: 3em;
    font-weight: bold;
    border: none;
}

/* Sidebar */

.sidebar .sidebar-content {
    background-color: #020617;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.markdown("""
<div class="enterprise-header">
<div class="header-title">
🏢 AutoML Studio — Enterprise Dashboard
</div>
<div class="header-subtitle">
Train • Compare • Deploy Machine Learning Models
</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.title("AutoML Studio")

file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("Dataset Loaded Successfully")

    # KPI METRICS

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_cols) > 0:

        col = st.selectbox(
            "Distribution Column",
            numeric_cols
        )

        st.plotly_chart(
            px.histogram(df, x=col),
            use_container_width=True
        )

    # ---------------- Preprocessing ----------------

    st.subheader("Preprocessing")

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
                df[col].astype(str)
            )

        st.session_state.df = df
        st.success("Encoding Applied")

    # ---------------- Scaling ----------------

    num_cols = df.select_dtypes(
        include=np.number
    ).columns

    scale_cols = st.multiselect(
        "Columns for Scaling",
        num_cols
    )

    scale_method = st.selectbox(
        "Scaling Method",
        [
            "Standardization",
            "Normalization"
        ]
    )

    if st.button("Apply Scaling"):

        scaler = StandardScaler() if scale_method=="Standardization" else MinMaxScaler()

        df[scale_cols] = scaler.fit_transform(
            df[scale_cols]
        )

        st.session_state.df = df
        st.success("Scaling Applied")

    learning_type = st.radio(
        "Select Learning Type",
        [
            "Supervised",
            "Unsupervised"
        ]
    )

else:

    st.info("Upload dataset to start AutoML")
