import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import time

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc
)

st.set_page_config(page_title="AutoML Studio", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a,#1e293b);color:white;}
.hero {background: linear-gradient(135deg,rgba(102,126,234,0.6),rgba(118,75,162,0.6));
padding:40px;border-radius:20px;}
.card {background: rgba(255,255,255,0.05);padding:20px;border-radius:15px;text-align:center;}
.upload-box {background: linear-gradient(135deg,#f6d365,#fda085);
padding:20px;border-radius:15px;text-align:center;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
<h1>🚀 AutoML Studio</h1>
<h3>Build ML Models in Seconds</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 📂 Upload Dataset")

file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# ============================
# NEW: LEARNING TYPE SELECTOR
# ============================

learning_type = st.sidebar.radio(
    "Select Learning Type",
    [
        "Supervised",
        "Unsupervised"
    ]
)

# ============================

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("Dataset Loaded")

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
            px.histogram(
                df,
                x=col
            )
        )

# ============================
# PREPROCESSING (COMMON)
# ============================

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

            if fill_method == "Mean":
                df[col] = df[col].fillna(
                    df[col].mean()
                )

            elif fill_method == "Median":
                df[col] = df[col].fillna(
                    df[col].median()
                )

            elif fill_method == "Mode":
                df[col] = df[col].fillna(
                    df[col].mode()[0]
                )

            elif fill_method == "Forward Fill":
                df[col] = df[col].ffill()

            elif fill_method == "Backward Fill":
                df[col] = df[col].bfill()

        st.success("Missing Values Handled")

# =====================================================
# SUPERVISED
# =====================================================

    if learning_type == "Supervised":

        st.subheader("Supervised Learning")

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

        st.write("Task:", task)

        X_train,X_test,y_train,y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        results = []

# ---------------- Classification ----------------

        if task == "Classification":

            best_score = 0

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier()
            }

            for name,model in models.items():

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

                results.append(
                    [
                        name,
                        acc
                    ]
                )

                if acc > best_score:

                    best_score = acc
                    best_model = model
                    best_name = name

            res = pd.DataFrame(
                results,
                columns=[
                    "Model",
                    "Accuracy"
                ]
            )

            st.dataframe(res)

            st.success(
                f"Best Model: {best_name}"
            )

# ---------------- Regression ----------------

        else:

            best_score = float("inf")

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor()
            }

            for name,model in models.items():

                model.fit(
                    X_train,
                    y_train
                )

                preds = model.predict(
                    X_test
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        preds
                    )
                )

                results.append(
                    [
                        name,
                        rmse
                    ]
                )

                if rmse < best_score:

                    best_score = rmse
                    best_model = model
                    best_name = name

            res = pd.DataFrame(
                results,
                columns=[
                    "Model",
                    "RMSE"
                ]
            )

            st.dataframe(res)

            st.success(
                f"Best Model: {best_name}"
            )

# =====================================================
# UNSUPERVISED
# =====================================================

    else:

        st.subheader("Unsupervised Learning")

        data = df.select_dtypes(
            include=np.number
        )

        algorithm = st.selectbox(
            "Clustering Algorithm",
            [
                "KMeans",
                "DBSCAN",
                "Hierarchical"
            ]
        )

        n_clusters = st.slider(
            "Number of Clusters",
            2,
            10,
            3
        )

        if st.button("Run Clustering"):

            if algorithm == "KMeans":

                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=42
                )

                labels = model.fit_predict(
                    data
                )

            elif algorithm == "DBSCAN":

                model = DBSCAN()

                labels = model.fit_predict(
                    data
                )

            else:

                model = AgglomerativeClustering(
                    n_clusters=n_clusters
                )

                labels = model.fit_predict(
                    data
                )

            df["Cluster"] = labels

            st.success("Clustering Completed")

            st.dataframe(
                df["Cluster"].value_counts()
            )

            pca = PCA(
                n_components=2
            )

            reduced = pca.fit_transform(
                data
            )

            plot_df = pd.DataFrame(
                reduced,
                columns=[
                    "PC1",
                    "PC2"
                ]
            )

            plot_df["Cluster"] = labels

            fig = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color="Cluster"
            )

            st.plotly_chart(fig)

else:

    st.markdown("""
    <div class="upload-box">
    Upload dataset to start
    </div>
    """, unsafe_allow_html=True)
