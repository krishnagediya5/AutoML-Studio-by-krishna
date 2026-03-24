# =========================================================
# AUTO ML STUDIO — DASHBOARD STYLE UI
# LOGIC: UNCHANGED
# UI: PROFESSIONAL DASHBOARD
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
    page_title="AutoML Studio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DASHBOARD CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 800;
}

.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR — DASHBOARD STYLE
# =========================================================

st.sidebar.title("📊 AutoML Dashboard")

st.sidebar.markdown("---")

st.sidebar.subheader("📂 Upload Dataset")

file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Workflow**

    1) Upload Dataset
    2) Preprocess Data
    3) Train Models
    4) View Results
    """
)

# =========================================================
# MAIN APP
# =========================================================

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    # =====================================================
    # HEADER
    # =====================================================

    col_title, col_space = st.columns([3,1])

    with col_title:
        st.markdown('<div class="dashboard-title">📊 AutoML Model Dashboard</div>', unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # METRICS ROW
    # =====================================================

    total_missing = int(df.isnull().sum().sum())

    m1, m2, m3 = st.columns(3)

    m1.metric("Rows", df.shape[0])
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing Values", total_missing)

    st.markdown("---")

    # =====================================================
    # DATA PREVIEW
    # =====================================================

    st.markdown('<div class="section-title">📄 Dataset Preview</div>', unsafe_allow_html=True)

    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        st.markdown('<div class="section-title">📈 Data Distribution</div>', unsafe_allow_html=True)

        col = st.selectbox("Select Column", numeric_cols)

        fig_hist = px.histogram(df, x=col)

        st.plotly_chart(fig_hist, use_container_width=True)

    # =====================================================
    # PREPROCESSING
    # =====================================================

    st.markdown('<div class="section-title">🧹 Data Preprocessing</div>', unsafe_allow_html=True)

    fill_cols = st.multiselect("Select Columns", df.columns)

    fill_method = st.selectbox(
        "Fill Method",
        ["Mean","Median","Mode","Forward Fill","Backward Fill"]
    )

    if st.button("Apply Missing Value Handling"):

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

        st.success("Missing Values Handled Successfully")

    # =====================================================
    # ENCODING
    # =====================================================

    cat_cols = df.select_dtypes(include="object").columns

    encode_cols = st.multiselect("Categorical Columns", cat_cols)

    if st.button("Apply Encoding"):

        for col in encode_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        st.session_state.df = df

        st.success("Encoding Applied")

    # =====================================================
    # SCALING
    # =====================================================

    num_cols = df.select_dtypes(include=np.number).columns

    scale_cols = st.multiselect("Scaling Columns", num_cols)

    scale_method = st.selectbox(
        "Scaling Method",
        ["Standardization","Normalization"]
    )

    if st.button("Apply Scaling"):

        scaler = StandardScaler() if scale_method=="Standardization" else MinMaxScaler()

        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        st.session_state.df = df

        st.success("Scaling Applied")

    st.markdown("---")

    learning_type = st.radio(
        "Select Learning Type",
        ["Supervised","Unsupervised"]
    )

    # =========================================================
    # SUPERVISED
    # =========================================================

    if learning_type == "Supervised":

        st.markdown('<div class="section-title">⚙️ Model Training</div>', unsafe_allow_html=True)

        target = st.selectbox("Target Column", df.columns)

        df = df.dropna(subset=[target])

        X = df.drop(columns=[target])
        y = df[target]

        target_type = type_of_target(y)

        if target_type in ["binary", "multiclass"]:
            task = "Classification"
        else:
            task = "Regression"

        st.info(f"Detected Task: {task}")

        k = st.slider("Top K Features",1,X.shape[1],min(5,X.shape[1]))

        selector = SelectKBest(
            f_classif if task=="Classification" else f_regression,
            k=k
        )

        X_new = selector.fit_transform(X,y)

        selected_features = X.columns[selector.get_support()]

        X = pd.DataFrame(X_new,columns=selected_features)

        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=0.2,random_state=42
        )

        st.markdown('<div class="section-title">🏆 Model Leaderboard</div>', unsafe_allow_html=True)

        results=[]
        best_model=None
        best_model_name=None

        progress = st.progress(0)

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

            for i,(name,model) in enumerate(models.items()):

                model.fit(X_train,y_train)

                preds=model.predict(X_test)

                acc=accuracy_score(y_test,preds)

                results.append([name,acc])

                progress.progress((i+1)/len(models))

                if acc>best_score:

                    best_score=acc
                    best_model=model
                    best_model_name=name

            res=pd.DataFrame(results,columns=["Model","Accuracy"])

            st.dataframe(res,use_container_width=True)

            st.success(f"Best Model: {best_model_name}")

        else:

            best_score=float("inf")

            models={
                "Linear Regression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "Random Forest":RandomForestRegressor(),
                "Extra Trees":ExtraTreesRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "KNN":KNeighborsRegressor(),
                "SVR":SVR()
            }

            for i,(name,model) in enumerate(models.items()):

                model.fit(X_train,y_train)

                preds=model.predict(X_test)

                rmse=np.sqrt(mean_squared_error(y_test,preds))

                results.append([name,rmse])

                progress.progress((i+1)/len(models))

                if rmse<best_score:

                    best_score=rmse
                    best_model=model
                    best_model_name=name

            res=pd.DataFrame(results,columns=["Model","RMSE"])

            st.dataframe(res,use_container_width=True)

            st.success(f"Best Model: {best_model_name}")

    # =========================================================
    # UNSUPERVISED
    # =========================================================

    else:

        st.markdown('<div class="section-title">🧠 Clustering Dashboard</div>', unsafe_allow_html=True)

        data = df.select_dtypes(include=np.number)

        scaler = StandardScaler()

        data_scaled = scaler.fit_transform(data)

        models = {
            "KMeans": KMeans(n_clusters=3),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
            "Birch": Birch(n_clusters=3),
            "DBSCAN": DBSCAN()
        }

        results=[]
        best_score=-1

        progress = st.progress(0)

        for i,(name,model) in enumerate(models.items()):

            labels = model.fit_predict(data_scaled)

            if len(set(labels)) > 1:

                score = silhouette_score(
                    data_scaled,
                    labels
                )

            else:

                score = -1

            results.append([name,score])

            progress.progress((i+1)/len(models))

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

        st.dataframe(res,use_container_width=True)

        st.success(
            f"Best Clustering Model: {best_model_name}"
        )

else:

    # =========================================================
    # DASHBOARD HOME PAGE
    # =========================================================

    st.markdown('<div class="dashboard-title">📊 AutoML Dashboard</div>', unsafe_allow_html=True)

    st.markdown("---")

    c1,c2,c3 = st.columns(3)

    c1.metric("Models", "15+")
    c2.metric("Algorithms", "Classification / Regression / Clustering")
    c3.metric("Automation", "100%")

    st.markdown("---")

    st.subheader("🚀 Platform Features")

    f1,f2,f3 = st.columns(3)

    f1.success("Upload Dataset")
    f2.info("Train Models Automatically")
    f3.warning("View Leaderboard & Predictions")

    st.markdown("---")

    st.info("Upload your dataset from the sidebar to start the AutoML dashboard.")
