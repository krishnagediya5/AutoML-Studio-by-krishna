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
    background: linear-gradient(180deg,#020617,#020617);
}

.hero {
    background: linear-gradient(135deg,#6366f1,#9333ea,#ec4899);
    padding: 48px;
    border-radius: 26px;
    margin-bottom: 28px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: #e5e7eb;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    color: white;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#0f172a);
}

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
AutoML Studio
</div>

<div class="hero-subtitle">
Train • Compare • Predict • Cluster — All in One Place
</div>

</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

c1.markdown('<div class="card">⚡ Fast Training</div>', unsafe_allow_html=True)
c2.markdown('<div class="card">🤖 Best Model</div>', unsafe_allow_html=True)
c3.markdown('<div class="card">📊 Analytics</div>', unsafe_allow_html=True)
c4.markdown('<div class="card">🧠 Clustering</div>', unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# FILE UPLOAD
# =====================================================

st.sidebar.markdown("## Upload Dataset")

file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("Dataset Loaded Successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =====================================================
    # PREPROCESSING
    # =====================================================

    st.subheader("Preprocessing")

    fill_cols = st.multiselect("Columns", df.columns)

    fill_method = st.selectbox(
        "Method",
        ["Mean","Median","Mode","Forward Fill","Backward Fill"]
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

        st.success("Missing Values Handled")

    # =====================================================
    # ENCODING
    # =====================================================

    cat_cols = df.select_dtypes(include="object").columns

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

    # =====================================================
    # SCALING
    # =====================================================

    num_cols = df.select_dtypes(include=np.number).columns

    scale_cols = st.multiselect(
        "Columns for Scaling",
        num_cols
    )

    scale_method = st.selectbox(
        "Scaling Method",
        ["Standardization","Normalization"]
    )

    if st.button("Apply Scaling"):

        scaler = (
            StandardScaler()
            if scale_method == "Standardization"
            else MinMaxScaler()
        )

        df[scale_cols] = scaler.fit_transform(
            df[scale_cols]
        )

        st.success("Scaling Applied")

    # =====================================================
    # LEARNING TYPE
    # =====================================================

    learning_type = st.radio(
        "Select Learning Type",
        ["Supervised", "Unsupervised"]
    )

    # =====================================================
    # SUPERVISED
    # =====================================================

    if learning_type == "Supervised":

        target = st.selectbox(
            "Select Target Column",
            df.columns
        )

        df = df.dropna(subset=[target])

        X = df.drop(columns=[target])
        y = df[target]

        target_type = type_of_target(y)

        if target_type in ["binary", "multiclass"]:
            task = "Classification"
        else:
            task = "Regression"

        st.subheader("Model Leaderboard")

        with st.spinner("Training models... Please wait ⏳"):

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

                for name, model in models.items():

                    model.fit(X_train, y_train)

                    preds = model.predict(X_test)

                    acc = accuracy_score(y_test, preds)

                    results.append([name, acc])

                    if acc > best_score:
                        best_score = acc
                        best_model = model
                        best_model_name = name

                leaderboard = pd.DataFrame(
                    results,
                    columns=["Model", "Accuracy"]
                ).sort_values(
                    by="Accuracy",
                    ascending=False
                )

                st.dataframe(leaderboard)

                st.success(
                    f"Best Model Selected: {best_model_name}"
                )

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
                        best_model_name = name

                leaderboard = pd.DataFrame(
                    results,
                    columns=["Model", "RMSE"]
                ).sort_values(
                    by="RMSE"
                )

                st.dataframe(leaderboard)

                st.success(
                    f"Best Model Selected: {best_model_name}"
                )

    # =====================================================
    # UNSUPERVISED
    # =====================================================

    else:

        st.subheader("Unsupervised Model Leaderboard")

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

        for name, model in models.items():

            labels = model.fit_predict(data_scaled)

            if len(set(labels)) > 1:
                score = silhouette_score(
                    data_scaled,
                    labels
                )
            else:
                score = -1

            results.append([name, score])

        leaderboard = pd.DataFrame(
            results,
            columns=["Algorithm", "Silhouette Score"]
        ).sort_values(
            by="Silhouette Score",
            ascending=False
        )

        st.dataframe(leaderboard)

        best_labels = labels

        pca = PCA(n_components=2)

        reduced = pca.fit_transform(data_scaled)

        plot_df = pd.DataFrame(
            reduced,
            columns=["PC1","PC2"]
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

    st.info("Upload dataset to start AutoML")

st.markdown(
    '<div class="footer">Developed by Krishna Gediya | AutoML Application</div>',
    unsafe_allow_html=True
)
