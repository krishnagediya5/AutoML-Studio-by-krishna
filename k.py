import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import time
import io

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
    mean_squared_error
)

st.set_page_config(page_title="AutoML Studio", layout="wide")

# ---------------- SIDEBAR ----------------

st.sidebar.markdown("## 📂 Upload Dataset")

file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("Dataset Loaded Successfully")

# ---------------- PREVIEW ----------------

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

# ---------------- INFO ----------------

    col1, col2 = st.columns(2)

    col1.write(f"Shape: {df.shape}")

    col2.write("Missing Values")

    col2.dataframe(
        df.isnull().sum().to_frame("Count")
    )

# ---------------- EDA ----------------

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_cols) > 0:

        col = st.selectbox(
            "Distribution Column",
            numeric_cols
        )

        fig_hist = px.histogram(
            df,
            x=col
        )

        st.plotly_chart(fig_hist)

# ---------------- PREPROCESSING ----------------

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
                df[col] = df[col].fillna(df[col].mean())

            elif fill_method == "Median":
                df[col] = df[col].fillna(df[col].median())

            elif fill_method == "Mode":
                df[col] = df[col].fillna(df[col].mode()[0])

            elif fill_method == "Forward Fill":
                df[col] = df[col].ffill()

            elif fill_method == "Backward Fill":
                df[col] = df[col].bfill()

        st.session_state.df = df

        st.success("Missing Values Handled")

# ---------------- LEARNING TYPE ----------------

    learning_type = st.radio(
        "Select Learning Type",
        [
            "Supervised",
            "Unsupervised"
        ]
    )

# =====================================================
# SUPERVISED
# =====================================================

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

        st.write(f"Task: {task}")

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

        st.subheader("Model Leaderboard")

        results = []

        best_model = None

        best_model_name = None

# ---------------- Classification ----------------

        if task == "Classification":

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

            for name, model in models.items():

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

                    best_model_name = name

            res = pd.DataFrame(
                results,
                columns=[
                    "Model",
                    "Accuracy"
                ]
            )

            st.dataframe(res)

            # NEW GRAPH

            fig_bar = px.bar(
                res,
                x="Model",
                y="Accuracy",
                title="Model Comparison"
            )

            st.plotly_chart(fig_bar)

            st.success(
                f"Best Model Selected: {best_model_name}"
            )

# ---------------- USER INPUT ----------------

        if best_model is not None:

            st.subheader("User Prediction")

            user_input = {}

            for col in X.columns:

                user_input[col] = st.number_input(
                    f"Enter value for {col}",
                    value=float(
                        X[col].mean()
                    )
                )

            input_df = pd.DataFrame(
                [user_input]
            )

            if st.button("Predict"):

                prediction = best_model.predict(
                    input_df
                )

                st.success(
                    f"Prediction Result: {prediction[0]}"
                )

            # DOWNLOAD MODEL

            buffer = io.BytesIO()

            pickle.dump(
                best_model,
                buffer
            )

            st.download_button(

                label="Download Best Model",

                data=buffer.getvalue(),

                file_name="best_model.pkl",

                mime="application/octet-stream"

            )

# =====================================================
# UNSUPERVISED
# =====================================================

    else:

        st.subheader(
            "Unsupervised Model Leaderboard"
        )

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

        best_score = -1

        for name, model in models.items():

            labels = model.fit_predict(
                data_scaled
            )

            if len(
                set(labels)
            ) > 1:

                score = silhouette_score(
                    data_scaled,
                    labels
                )

            else:

                score = -1

            results.append(
                [
                    name,
                    score
                ]
            )

            if score > best_score:

                best_score = score

                best_model = model

                best_model_name = name

                best_labels = labels

        res = pd.DataFrame(
            results,
            columns=[
                "Algorithm",
                "Silhouette Score"
            ]
        )

        st.dataframe(res)

        st.success(
            f"Best Clustering Model: {best_model_name}"
        )

        df["Cluster"] = best_labels

        pca = PCA(
            n_components=2
        )

        reduced = pca.fit_transform(
            data_scaled
        )

        plot_df = pd.DataFrame(
            reduced,
            columns=[
                "PC1",
                "PC2"
            ]
        )

        plot_df["Cluster"] = best_labels

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title="Cluster Visualization"
        )

        st.plotly_chart(fig)

else:

    st.info(
        "Upload dataset to start AutoML"
    )
