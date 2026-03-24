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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS (UI IMPROVEMENT ONLY — NO LOGIC CHANGE)
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }

    .metric-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    .section-header {
        font-size: 26px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🤖 AutoML Studio")
st.sidebar.markdown("---")

st.sidebar.markdown("## 📂 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Workflow**

    1) Upload dataset
    2) Preprocess data
    3) Train models
    4) Get predictions
    """
)

# =========================================================
# MAIN APP
# =========================================================

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("✅ Dataset Loaded Successfully")

    # =====================================================
    # DATA PREVIEW
    # =====================================================

    st.markdown("<div class='section-header'>📊 Dataset Overview</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        st.markdown("<div class='section-header'>📈 Data Distribution</div>", unsafe_allow_html=True)

        col = st.selectbox("Select Column", numeric_cols)

        fig_hist = px.histogram(
            df,
            x=col,
            title="Distribution"
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    # =====================================================
    # PREPROCESSING
    # =====================================================

    st.markdown("<div class='section-header'>🧹 Preprocessing</div>", unsafe_allow_html=True)

    fill_cols = st.multiselect("Columns", df.columns)

    fill_method = st.selectbox(
        "Method",
        ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill"]
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
        st.success("✅ Missing Values Handled")

    # =====================================================
    # ENCODING
    # =====================================================

    cat_cols = df.select_dtypes(include="object").columns

    encode_cols = st.multiselect("Categorical Columns", cat_cols)

    if st.button("Apply Encoding"):

        for col in encode_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        st.session_state.df = df
        st.success("✅ Encoding Applied")

    # =====================================================
    # SCALING
    # =====================================================

    num_cols = df.select_dtypes(include=np.number).columns

    scale_cols = st.multiselect("Columns for Scaling", num_cols)

    scale_method = st.selectbox(
        "Scaling Method",
        ["Standardization", "Normalization"]
    )

    if st.button("Apply Scaling"):

        scaler = StandardScaler() if scale_method == "Standardization" else MinMaxScaler()

        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        st.session_state.df = df
        st.success("✅ Scaling Applied")

    st.markdown("---")

    learning_type = st.radio(
        "🧠 Select Learning Type",
        ["Supervised", "Unsupervised"]
    )

    # =========================================================
    # SUPERVISED
    # =========================================================

    if learning_type == "Supervised":

        st.markdown("<div class='section-header'>⚙️ Model Setup</div>", unsafe_allow_html=True)

        target = st.selectbox("Target Column", df.columns)

        df = df.dropna(subset=[target])

        X = df.drop(columns=[target])
        y = df[target]

        target_type = type_of_target(y)

        if target_type in ["binary", "multiclass"]:
            task = "Classification"
        else:
            task = "Regression"

        st.info(f"🎯 Task Detected: {task}")

        k = st.slider("Top K Features", 1, X.shape[1], min(5, X.shape[1]))

        selector = SelectKBest(
            f_classif if task == "Classification" else f_regression,
            k=k
        )

        X_new = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()]

        X = pd.DataFrame(X_new, columns=selected_features)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.markdown("<div class='section-header'>🏆 Model Leaderboard</div>", unsafe_allow_html=True)

        results = []
        best_model = None
        best_model_name = None

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

            progress = st.progress(0)

            for i, (name, model) in enumerate(models.items()):

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)

                results.append([name, acc])

                progress.progress((i + 1) / len(models))

                if acc > best_score:

                    best_score = acc
                    best_model = model
                    best_model_name = name

            res = pd.DataFrame(results, columns=["Model", "Accuracy"])

            st.dataframe(res, use_container_width=True)

            st.success(f"Best Model Selected: {best_model_name}")

            if hasattr(best_model, "feature_importances_"):

                st.markdown("<div class='section-header'>⭐ Feature Importance</div>", unsafe_allow_html=True)

                importance = best_model.feature_importances_

                fi_df = pd.DataFrame({
                    "Feature": selected_features,
                    "Importance": importance
                })

                fig_imp = px.bar(
                    fi_df,
                    x="Feature",
                    y="Importance",
                    title="Feature Importance"
                )

                st.plotly_chart(fig_imp, use_container_width=True)

        else:

            best_score = float("inf")

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

            progress = st.progress(0)

            for i, (name, model) in enumerate(models.items()):

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, preds))

                results.append([name, rmse])

                progress.progress((i + 1) / len(models))

                if rmse < best_score:

                    best_score = rmse
                    best_model = model
                    best_model_name = name

            res = pd.DataFrame(results, columns=["Model", "RMSE"])

            st.dataframe(res, use_container_width=True)

            st.success(f"Best Model Selected: {best_model_name}")

            if hasattr(best_model, "feature_importances_"):

                st.markdown("<div class='section-header'>⭐ Feature Importance</div>", unsafe_allow_html=True)

                importance = best_model.feature_importances_

                fi_df = pd.DataFrame({
                    "Feature": selected_features,
                    "Importance": importance
                })

                fig_imp = px.bar(
                    fi_df,
                    x="Feature",
                    y="Importance",
                    title="Feature Importance"
                )

                st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("<div class='section-header'>🧑‍💻 User Input Prediction</div>", unsafe_allow_html=True)

            user_data = {}

            for col in selected_features:
                val = st.number_input(
                    f"Enter value for {col}",
                    value=0.0
                )
                user_data[col] = val

            if st.button("Predict"):

                input_df = pd.DataFrame([user_data])

                prediction = best_model.predict(input_df)

                st.success(f"Prediction: {prediction[0]}")

    # =========================================================
    # UNSUPERVISED
    # =========================================================

    else:

        st.markdown("<div class='section-header'>🧠 Unsupervised Model Leaderboard</div>", unsafe_allow_html=True)

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

            progress.progress((i + 1) / len(models))

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

        st.dataframe(res, use_container_width=True)

        st.success(
            f"Best Clustering Model: {best_model_name}"
        )

        df["Cluster"] = best_labels

        pca = PCA(n_components=2)

        reduced = pca.fit_transform(data_scaled)

        plot_df = pd.DataFrame(
            reduced,
            columns=["PC1", "PC2"]
        )

        plot_df["Cluster"] = best_labels

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title="Cluster Visualization"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>⭐ Feature Importance (Unsupervised)</div>", unsafe_allow_html=True)

        pca_imp = PCA(n_components=2)
        pca_imp.fit(data_scaled)

        importance_values = np.mean(
            np.abs(pca_imp.components_),
            axis=0
        )

        fi_unsup_df = pd.DataFrame({
            "Feature": data.columns,
            "Importance": importance_values
        })

        fig_unsup_imp = px.bar(
            fi_unsup_df,
            x="Feature",
            y="Importance",
            title="Unsupervised Feature Importance"
        )

        st.plotly_chart(fig_unsup_imp, use_container_width=True)

        st.markdown("<div class='section-header'>🧑‍💻 User Input Cluster Prediction</div>", unsafe_allow_html=True)

        user_data = {}

        for col in data.columns:

            val = st.number_input(
                f"Enter value for {col}",
                value=0.0,
                key=f"unsup_{col}"
            )

            user_data[col] = val

        if st.button("Predict Cluster"):

            input_df = pd.DataFrame(
                [user_data]
            )

            input_scaled = scaler.transform(
                input_df
            )

            if best_model_name == "KMeans":

                model = KMeans(
                    n_clusters=3
                )

                model.fit(data_scaled)

                cluster = model.predict(
                    input_scaled
                )

                st.success(
                    f"Predicted Cluster: {cluster[0]}"
                )

            elif best_model_name == "Birch":

                model = Birch(
                    n_clusters=3
                )

                model.fit(data_scaled)

                cluster = model.predict(
                    input_scaled
                )

                st.success(
                    f"Predicted Cluster: {cluster[0]}"
                )

            else:

                st.warning(
                    "Prediction not supported for this clustering algorithm"
                )

else:

    # =========================================================
    # FRONT PAGE (UI ONLY — LOGIC SAME)
    # =========================================================

    st.markdown("<div class='main-title'>🤖 AutoML Studio</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='subtitle'>Build, Train, and Evaluate Machine Learning Models Automatically</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("🚀 Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
            📂 Upload Dataset

            - CSV support
            - Data preview
            - Missing value handling
            """
        )

    with col2:
        st.success(
            """
            🤖 Train Models

            - Classification
            - Regression
            - Clustering
            """
        )

    with col3:
        st.warning(
            """
            📊 Get Results

            - Leaderboard
            - Feature importance
            - Predictions
            """
        )

    st.markdown("---")

    st.subheader("⚙️ How It Works")

    step1, step2, step3 = st.columns(3)

    step1.metric("Step 1", "Upload Dataset")
    step2.metric("Step 2", "Train Model")
    step3.metric("Step 3", "Get Results")

    st.markdown("---")

    st.info("👈 Upload your dataset using the sidebar to start AutoML.")

    st.markdown("---")

    st.caption("AutoML Studio | Improved UI — Logic Unchanged")
