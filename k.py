import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
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
from sklearn.metrics import silhouette_score, accuracy_score, mean_squared_error
from sklearn.decomposition import PCA


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# STYLE
# =====================================================

st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at 20% 20%, #0f172a, #020617);
}

.hero {
    background: linear-gradient(135deg,#2563eb,#7c3aed,#9333ea);
    padding: 60px;
    border-radius: 24px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: #e2e8f0;
}

.metric-card {
    background: rgba(255,255,255,0.06);
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: white;
}

.metric-label {
    font-size: 14px;
    color: #94a3b8;
}

.card {
    background: rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    color: white;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#0f172a);
}

.footer {
    margin-top: 50px;
    padding: 18px;
    text-align: center;
    color: #9ca3af;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🚀 AutoML Studio")

file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# =====================================================
# HERO
# =====================================================

st.markdown("""
<div class="hero">
<div class="hero-title">🚀 AutoML Studio</div>
<div class="hero-subtitle">
Automated Machine Learning • Smart Model Selection • Fast Predictions
</div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# PRODUCT DESCRIPTION
# =====================================================

st.markdown("""
### ⚡ Built for Speed. Designed for Intelligence.

Upload your dataset and instantly train multiple machine learning models.  
Compare performance, select the best model, and visualize results — automatically.
""")

# =====================================================
# MAIN LOGIC
# =====================================================

if file is not None:

    df = pd.read_csv(file)

    st.success("Dataset Loaded Successfully")

    # =====================================================
    # DASHBOARD CARDS
    # =====================================================

    rows = df.shape[0]
    cols = df.shape[1]
    missing = df.isnull().sum().sum()
    memory = round(df.memory_usage().sum() / 1024, 2)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">{rows}</div>
        <div class="metric-label">Rows</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">{cols}</div>
        <div class="metric-label">Columns</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">{missing}</div>
        <div class="metric-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">{memory} KB</div>
        <div class="metric-label">Memory Usage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # =========================
    # PREPROCESSING
    # =========================

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

        st.success("Missing Values Handled")

    # =========================
    # ENCODING
    # =========================

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

        st.success("Encoding Applied")

    # =========================
    # SCALING
    # =========================

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

        scaler = (
            StandardScaler()
            if scale_method == "Standardization"
            else MinMaxScaler()
        )

        if len(scale_cols) > 0:

            df[scale_cols] = scaler.fit_transform(
                df[scale_cols]
            )

        st.success("Scaling Applied")

    # =========================
    # LEARNING TYPE
    # =========================

    learning_type = st.radio(
        "Select Learning Type",
        [
            "Supervised",
            "Unsupervised"
        ]
    )

    if learning_type == "Supervised":

        target = st.selectbox(
            "Select Target Column",
            df.columns
        )

        df = df.dropna(
            subset=[target]
        )

        X = df.drop(columns=[target])
        y = df[target]

        target_type = type_of_target(y)

        if target_type in ["binary","multiclass"]:
            task = "Classification"
        else:
            task = "Regression"

        st.info(f"Detected Task Type: {task}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        results = []

        if task == "Classification":

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

            best_score = 0
            best_model_name = None

            for name, model in models.items():

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)

                results.append([name, acc])

                if acc > best_score:
                    best_score = acc
                    best_model_name = name

            leaderboard = pd.DataFrame(
                results,
                columns=["Model","Accuracy"]
            ).sort_values(
                by="Accuracy",
                ascending=False
            )

            st.dataframe(leaderboard)

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
            best_model_name = None

            for name, model in models.items():

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        preds
                    )
                )

                results.append([name, rmse])

                if rmse < best_score:
                    best_score = rmse
                    best_model_name = name

            leaderboard = pd.DataFrame(
                results,
                columns=["Model","RMSE"]
            ).sort_values(
                by="RMSE"
            )

            st.dataframe(leaderboard)

            st.success(f"Best Model: {best_model_name}")

else:

    st.info("Upload dataset to start AutoML")

st.markdown(
    '<div class="footer">Developed by Krishna Gediya | AutoML System</div>',
    unsafe_allow_html=True
)
