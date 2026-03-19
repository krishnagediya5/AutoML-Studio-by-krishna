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

from sklearn.metrics import *

st.set_page_config(page_title="AutoML Studio", layout="wide")

# ---------------- GLASS UI ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.hero {
    background: linear-gradient(135deg, rgba(102,126,234,0.6), rgba(118,75,162,0.6));
    backdrop-filter: blur(20px);
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    animation: fadeIn 0.8s ease-in-out;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    text-align:center;
}

.upload-box {
    background: linear-gradient(135deg, #f6d365, #fda085);
    padding: 20px;
    border-radius: 15px;
    text-align:center;
    font-size:18px;
    font-weight:600;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.3);
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(25px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
<h1>🚀 AutoML Studio</h1>
<h3>🔥 Build Machine Learning Models in Seconds</h3>
<p>Upload datasets, preprocess data, train models & predict — all in one place.</p>
<p>👉 No coding • No complexity • Just powerful AI</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- CARDS ----------------
col1, col2, col3 = st.columns(3)
col1.markdown('<div class="card">⚡ Instant Speed</div>', unsafe_allow_html=True)
col2.markdown('<div class="card">🤖 Multiple Models</div>', unsafe_allow_html=True)
col3.markdown('<div class="card">📊 Optimized Results</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 📂 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------------- MAIN ----------------
if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("✅ Dataset Loaded Successfully")

# ---------------- PREVIEW ----------------
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

# ---------------- INFO ----------------
    col1, col2 = st.columns(2)
    col1.write("📐 Shape:", df.shape)
    col2.write(df.isnull().sum())

# ---------------- EDA ----------------
    num_cols = df.select_dtypes(include=np.number).columns

    if len(num_cols) > 0:
        col = st.selectbox("📈 Distribution Column", num_cols)
        st.plotly_chart(px.histogram(df, x=col))

# ---------------- PREPROCESS ----------------
    st.subheader("🧹 Preprocessing")

    fill_cols = st.multiselect("Columns", df.columns)
    method = st.selectbox("Method", ["Mean","Median","Mode"])

    if st.button("Apply Missing Fill"):
        for col in fill_cols:
            if method=="Mean":
                df[col]=df[col].fillna(df[col].mean())
            elif method=="Median":
                df[col]=df[col].fillna(df[col].median())
            else:
                df[col]=df[col].fillna(df[col].mode()[0])

        st.session_state.df=df
        st.success("Done")

# ---------------- MODEL ----------------
    st.subheader("🤖 Model")

    target = st.selectbox("Target", df.columns)

    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include="object"):
        X[col] = LabelEncoder().fit_transform(X[col])

    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    task = "Classification" if type_of_target(y) in ["binary","multiclass"] else "Regression"
    st.write("🎯 Task:", task)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    results=[]
    best_model=None
    best_model_name=None

# ---------------- TRAIN ----------------
    if task=="Classification":

        best_score=0
        models={
            "LR":LogisticRegression(max_iter=1000),
            "RF":RandomForestClassifier(),
            "KNN":KNeighborsClassifier()
        }

        with st.spinner("🤖 Training Models..."):
            progress=st.progress(0)
            total=len(models)

            for i,(name,m) in enumerate(models.items()):
                m.fit(X_train,y_train)
                acc=accuracy_score(y_test,m.predict(X_test))
                results.append([name,acc])

                if acc>best_score:
                    best_score=acc
                    best_model=m
                    best_model_name=name

                progress.progress((i+1)/total)
                time.sleep(0.2)

        st.success("✅ Training Completed")

        res=pd.DataFrame(results,columns=["Model","Accuracy"])
        st.dataframe(res)
        st.plotly_chart(px.bar(res,x="Model",y="Accuracy"))

# ---------------- REGRESSION ----------------
    else:

        best_score=float("inf")
        models={
            "LR":LinearRegression(),
            "RF":RandomForestRegressor(),
            "KNN":KNeighborsRegressor()
        }

        with st.spinner("🤖 Training Models..."):
            progress=st.progress(0)
            total=len(models)

            for i,(name,m) in enumerate(models.items()):
                m.fit(X_train,y_train)
                rmse=np.sqrt(mean_squared_error(y_test,m.predict(X_test)))
                results.append([name,rmse])

                if rmse<best_score:
                    best_score=rmse
                    best_model=m
                    best_model_name=name

                progress.progress((i+1)/total)
                time.sleep(0.2)

        st.success("✅ Training Completed")

        res=pd.DataFrame(results,columns=["Model","RMSE"])
        st.dataframe(res)
        st.plotly_chart(px.bar(res,x="Model",y="RMSE"))

# ---------------- PREDICT ----------------
    st.subheader("🔮 Prediction")

    user_input={}
    for col in X.columns:
        user_input[col]=st.number_input(col, value=float(X[col].mean()))

    input_df=pd.DataFrame([user_input])

    if st.button("Predict"):
        with st.spinner("🔮 Generating Prediction..."):
            time.sleep(1)
            pred=best_model.predict(input_df)

        st.success(f"Prediction: {pred[0]}")

else:

    st.markdown("""
    <div class="upload-box">
    📂 Upload your dataset to start building ML models 🚀
    </div>
    """, unsafe_allow_html=True)
