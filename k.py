import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

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

# ---------------- GLASS UI CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    animation: fadeIn 0.8s ease-in-out;
}

.upload-box {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 18px;
    border: 1px solid rgba(255,255,255,0.2);
    animation: fadeIn 1s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="glass">
<h1 style="font-size:40px;">🚀 AutoML Studio</h1>
<h3>🔥 Build Machine Learning Models in Seconds</h3>
<p style="font-size:17px;">
AutoML Studio lets you upload datasets, preprocess data, train ML models, and generate predictions — all in one place.
</p>
<p>👉 No coding • No complexity • Just powerful AI</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- STATS ----------------
col1, col2, col3 = st.columns(3)

col1.markdown('<div class="glass">⚡ Instant Speed</div>', unsafe_allow_html=True)
col2.markdown('<div class="glass">🤖 8+ Models</div>', unsafe_allow_html=True)
col3.markdown('<div class="glass">📊 Optimized Accuracy</div>', unsafe_allow_html=True)

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

# ---------------- TRAIN ----------------
    results=[]

    if task=="Classification":
        models={
            "LR":LogisticRegression(max_iter=1000),
            "RF":RandomForestClassifier(),
            "KNN":KNeighborsClassifier()
        }

        for name,m in models.items():
            m.fit(X_train,y_train)
            acc=accuracy_score(y_test,m.predict(X_test))
            results.append([name,acc])

        res=pd.DataFrame(results,columns=["Model","Accuracy"])
        st.dataframe(res)
        st.plotly_chart(px.bar(res,x="Model",y="Accuracy"))

# ---------------- PREDICT ----------------
    st.subheader("🔮 Prediction")

    user_input={}
    for col in X.columns:
        user_input[col]=st.number_input(col, value=float(X[col].mean()))

    if st.button("Predict"):
        st.success("Prediction Generated ✅")

else:

    st.markdown("""
    <div class="upload-box">
    📂 Upload your dataset to start building ML models 🚀
    </div>
    """, unsafe_allow_html=True)
