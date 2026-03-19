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

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import *

st.set_page_config(page_title="AutoML Studio", layout="wide")

# ---------------- HERO ----------------
st.markdown("""
<div style="background: linear-gradient(135deg,#667eea,#764ba2);
padding:30px;border-radius:15px;color:white">
<h1>🚀 AutoML Studio</h1>
<p>End-to-End Machine Learning Platform — Build, Train & Predict Instantly</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["📂 Upload","📊 EDA","🧹 Preprocessing","🤖 Model","🔮 Prediction"])

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------------- LOAD ----------------
if file:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

# ---------------- DASHBOARD CARDS ----------------
    st.markdown("### 📊 Dashboard Overview")

    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

# ---------------- UPLOAD ----------------
    if page == "📂 Upload":
        st.subheader("📂 Dataset")
        st.dataframe(df.head())

# ---------------- EDA ----------------
    if page == "📊 EDA":
        st.subheader("📊 Data Visualization")

        num_cols = df.select_dtypes(include=np.number).columns
        col = st.selectbox("Select Feature", num_cols)

        fig1 = px.histogram(df, x=col)
        fig2 = px.box(df, y=col)

        c1,c2 = st.columns(2)
        c1.plotly_chart(fig1, use_container_width=True)
        c2.plotly_chart(fig2, use_container_width=True)

# ---------------- PREPROCESSING ----------------
    if page == "🧹 Preprocessing":

        st.subheader("🧹 Data Cleaning")

        cols = st.multiselect("Columns", df.columns)
        method = st.selectbox("Method", ["Mean","Median","Mode"])

        if st.button("Apply Cleaning"):
            for col in cols:
                if method=="Mean":
                    df[col]=df[col].fillna(df[col].mean())
                elif method=="Median":
                    df[col]=df[col].fillna(df[col].median())
                else:
                    df[col]=df[col].fillna(df[col].mode()[0])

            st.session_state.df=df
            st.success("Preprocessing Done ✅")

# ---------------- MODEL ----------------
    if page == "🤖 Model":

        st.subheader("🤖 Model Training")

        target = st.selectbox("Target Column", df.columns)

        df = df.dropna(subset=[target])
        X = df.drop(columns=[target])
        y = df[target]

        # encoding
        for col in X.select_dtypes(include="object"):
            X[col] = LabelEncoder().fit_transform(X[col])

        # scaling
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        task = "Classification" if type_of_target(y) in ["binary","multiclass"] else "Regression"

        st.success(f"Detected Task: {task}")

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

        results=[]

        if task=="Classification":
            models={
                "Logistic":LogisticRegression(max_iter=1000),
                "RandomForest":RandomForestClassifier(),
                "KNN":KNeighborsClassifier()
            }

            for name,m in models.items():
                m.fit(X_train,y_train)
                acc=accuracy_score(y_test,m.predict(X_test))
                results.append([name,acc])

            res=pd.DataFrame(results,columns=["Model","Accuracy"])
            st.dataframe(res)

            st.plotly_chart(px.bar(res,x="Model",y="Accuracy"))

# ---------------- PREDICTION ----------------
    if page == "🔮 Prediction":

        st.subheader("🔮 Prediction")

        st.info("Train model first from Model tab")

else:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#f6d365,#fda085);
    padding:25px;border-radius:10px;text-align:center;font-size:18px">
    🚀 Upload Dataset to Start AutoML Studio
    </div>
    """, unsafe_allow_html=True)
