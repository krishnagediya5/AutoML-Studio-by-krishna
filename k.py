import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

# =====================================================
# PAGE CONFIG (MUST BE FIRST)
# =====================================================

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# THEME TOGGLE
# =====================================================

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

with st.sidebar:
    theme_toggle = st.toggle("Dark Mode")

    if theme_toggle:
        st.session_state.theme_mode = "Dark"
    else:
        st.session_state.theme_mode = "Light"

# =====================================================
# STYLING
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg,#f8fafc,#eef2ff);
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

.feature-box {
    background: white;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
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
Train, Compare, and Deploy Machine Learning Models - No Code Required
</div>

</div>
""", unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)

f1.markdown('<div class="feature-box">Fast Training</div>', unsafe_allow_html=True)
f2.markdown('<div class="feature-box">Auto Model Selection</div>', unsafe_allow_html=True)
f3.markdown('<div class="feature-box">Smart Analytics</div>', unsafe_allow_html=True)
f4.markdown('<div class="feature-box">Cloud Ready</div>', unsafe_allow_html=True)

# =====================================================
# DATA UPLOAD
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

    col1, col2 = st.columns(2)

    col1.write(f"Shape: {df.shape}")

    col2.write("Missing Values")
    col2.dataframe(df.isnull().sum().to_frame("Count"))

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        col = st.selectbox(
            "Distribution Column",
            numeric_cols
        )

        st.plotly_chart(
            px.histogram(df, x=col)
        )

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

    scale_cols = st.multiselect("Columns for Scaling", num_cols)

    scale_method = st.selectbox(
        "Scaling Method",
        ["Standardization","Normalization"]
    )

    if st.button("Apply Scaling"):

        scaler = StandardScaler() if scale_method=="Standardization" else MinMaxScaler()

        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        st.session_state.df = df
        st.success("Scaling Applied")

    learning_type = st.radio(
        "Select Learning Type",
        ["Supervised","Unsupervised"]
    )

    # =====================================================
    # SUPERVISED
    # =====================================================

    if learning_type == "Supervised":

        st.subheader("Model Setup")

        target = st.selectbox("Target Column", df.columns)

        df = df.dropna(subset=[target])

        X = df.drop(columns=[target])
        y = df[target]

        target_type = type_of_target(y)

        if target_type in ["binary", "multiclass"]:
            task = "Classification"
        else:
            task = "Regression"

        st.write(f"Task: {task}")

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

        st.subheader("Model Leaderboard")

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

                model.fit(X_train,y_train)

                preds=model.predict(X_test)

                acc=accuracy_score(y_test,preds)

                results.append([name,acc])

                if acc>best_score:

                    best_score=acc
                    best_model=model
                    best_model_name=name

            res=pd.DataFrame(results,columns=["Model","Accuracy"])

            st.dataframe(res)

            st.success(f"Best Model Selected: {best_model_name}")

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

            for name,model in models.items():

                model.fit(X_train,y_train)

                preds=model.predict(X_test)

                rmse=np.sqrt(mean_squared_error(y_test,preds))

                results.append([name,rmse])

                if rmse<best_score:

                    best_score=rmse
                    best_model=model
                    best_model_name=name

            res=pd.DataFrame(results,columns=["Model","RMSE"])

            st.dataframe(res)

            st.success(f"Best Model Selected: {best_model_name}")

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

        results=[]
        best_score=-1

        for name,model in models.items():

            labels = model.fit_predict(data_scaled)

            if len(set(labels)) > 1:

                score = silhouette_score(
                    data_scaled,
                    labels
                )

            else:

                score = -1

            results.append([name,score])

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

        st.dataframe(res)

        st.success(
            f"Best Clustering Model: {best_model_name}"
        )

else:

    st.info("Upload dataset to start AutoML")
