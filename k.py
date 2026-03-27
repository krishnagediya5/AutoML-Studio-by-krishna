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
    page_title="AutoML Studio",
    page_icon="🚀",
    layout="wide"
)

# ----------------------------------------------------
# SaaS DASHBOARD UI
# ----------------------------------------------------

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.block-container {
    padding-top: 1rem;
}

.header-bar {
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.header-title {
    font-size: 34px;
    font-weight: bold;
    color: white;
}

.metric-card {
    background: #1e293b;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}

.sidebar .sidebar-content {
    background-color: #020617;
}

.stButton>button {
    background: linear-gradient(90deg,#2563eb,#22c55e);
    color: white;
    border-radius: 8px;
    height: 3em;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.markdown("""
<div class="header-bar">
<div class="header-title">
🚀 AutoML Studio — SaaS Dashboard
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

    # ----------------------------------------------------
    # KPI METRICS
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # SUPERVISED
    # ----------------------------------------------------

    if learning_type == "Supervised":

        st.subheader("Model Setup")

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

        target_type = type_of_target(
            y
        )

        if target_type in [
            "binary",
            "multiclass"
        ]:

            task = "Classification"

        else:

            task = "Regression"

        st.write(
            f"Task: {task}"
        )

        k = st.slider(
            "Top K Features",
            1,
            X.shape[1],
            min(
                5,
                X.shape[1]
            )
        )

        selector = SelectKBest(
            f_classif if task=="Classification" else f_regression,
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

        X_train,X_test,y_train,y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        st.subheader(
            "Model Leaderboard"
        )

        results=[]
        best_model=None
        best_model_name=None

        if task=="Classification":

            best_score=0

            models={

                "Logistic Regression":LogisticRegression(max_iter=1000),
                "Random Forest":RandomForestClassifier(),
                "Extra Trees":ExtraTreesClassifier(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "KNN":KNeighborsClassifier(),
                "SVM":SVC(probability=True),
                "Naive Bayes":GaussianNB()

            }

            for name,model in models.items():

                model.fit(
                    X_train,
                    y_train
                )

                preds=model.predict(
                    X_test
                )

                acc=accuracy_score(
                    y_test,
                    preds
                )

                results.append(
                    [name,acc]
                )

                if acc>best_score:

                    best_score=acc
                    best_model=model
                    best_model_name=name

            res=pd.DataFrame(
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

else:

    st.info(
        "Upload dataset to start AutoML"
    )
