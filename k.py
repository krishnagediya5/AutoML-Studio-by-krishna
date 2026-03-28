import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.decomposition import PCA

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🚀",
    layout="wide"
)

# ----------------------------------------------------
# UI
# ----------------------------------------------------

st.markdown("""
<style>
.feature-box {
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    padding: 16px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-weight: 600;
}

.hero {
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    padding: 36px;
    border-radius: 16px;
    margin-bottom: 28px;
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="hero-title">🚀 AutoML Studio</div>
<div class="hero-subtitle">Train, Compare, and Deploy Models — No Code Required</div>
</div>
""", unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)

f1.markdown('<div class="feature-box">⚡ Fast Training</div>', unsafe_allow_html=True)
f2.markdown('<div class="feature-box">🤖 Auto Model Selection</div>', unsafe_allow_html=True)
f3.markdown('<div class="feature-box">📊 Smart Analytics</div>', unsafe_allow_html=True)
f4.markdown('<div class="feature-box">☁️ Cloud Ready</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------

st.sidebar.markdown("## 📂 Upload Dataset")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.success("Dataset Loaded Successfully")

    st.dataframe(df.head())

    learning_type = st.radio("Select Learning Type", ["Supervised", "Unsupervised"])

    # =====================================================
    # SUPERVISED
    # =====================================================

    if learning_type == "Supervised":

        target = st.selectbox("Target Column", df.columns)

        df = df.dropna(subset=[target])

        X = df.drop(columns=[target])
        y = df[target]

        # SAFETY CHECKS

        if X.shape[1] == 0:
            st.error("No features available.")
            st.stop()

        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        target_type = type_of_target(y)

        task = "Classification" if target_type in ["binary", "multiclass"] else "Regression"

        st.write("Task:", task)

        max_features = max(1, X.shape[1])

        k = st.slider(
            "Top K Features",
            min_value=1,
            max_value=max_features,
            value=min(5, max_features)
        )

        selector = SelectKBest(
            f_classif if task == "Classification" else f_regression,
            k=k
        )

        X_new = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()]

        X = pd.DataFrame(X_new, columns=selected_features)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []

        if task == "Classification":

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

            best_score = 0

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                results.append([name, acc])

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_model_name = name

            st.dataframe(pd.DataFrame(results, columns=["Model", "Accuracy"]))
            st.success(f"Best Model: {best_model_name}")

        else:

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest": RandomForestRegressor(),
                "Extra Trees": ExtraTreesRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR()
            }

            best_score = float("inf")

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))

                results.append([name, rmse])

                if rmse < best_score:
                    best_score = rmse
                    best_model = model
                    best_model_name = name

            st.dataframe(pd.DataFrame(results, columns=["Model", "RMSE"]))
            st.success(f"Best Model: {best_model_name}")

    # =====================================================
    # UNSUPERVISED
    # =====================================================

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

        for name, model in models.items():

            labels = model.fit_predict(data_scaled)

            if len(set(labels)) > 1:
                score = silhouette_score(data_scaled, labels)
            else:
                score = -1

            results.append([name, score])

            if score > best_score:
                best_score = score
                best_model_name = name

        st.dataframe(pd.DataFrame(results, columns=["Algorithm", "Score"]))
        st.success(f"Best Model: {best_model_name}")

else:
    st.info("Upload dataset to start AutoML")
