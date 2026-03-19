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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc
)

st.set_page_config(page_title="AutoML Studio", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a,#1e293b);color:white;}
.hero {background: linear-gradient(135deg,rgba(102,126,234,0.6),rgba(118,75,162,0.6));
padding:40px;border-radius:20px;box-shadow:0 10px 40px rgba(0,0,0,0.4);}
.card {background: rgba(255,255,255,0.05);padding:20px;border-radius:15px;text-align:center;}
.upload-box {background: linear-gradient(135deg,#f6d365,#fda085);
padding:20px;border-radius:15px;text-align:center;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
<h1>🚀 AutoML Studio</h1>
<h3>🔥 Build Machine Learning Models in Seconds</h3>
<p>Upload datasets, preprocess, train & predict — all in one place.</p>
<p>👉 No coding • No complexity • Just powerful AI</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- CARDS ----------------
c1,c2,c3 = st.columns(3)
c1.markdown('<div class="card">⚡ Instant</div>', unsafe_allow_html=True)
c2.markdown('<div class="card">🤖 8+ Models</div>', unsafe_allow_html=True)
c3.markdown('<div class="card">📊 Optimized</div>', unsafe_allow_html=True)

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
    col1.write(f"📐 Shape: {df.shape}")
    col2.write("❗ Missing Values")
    col2.dataframe(df.isnull().sum().to_frame("Count"))

# ---------------- EDA ----------------
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        col = st.selectbox("📈 Distribution Column", numeric_cols)
        st.plotly_chart(px.histogram(df, x=col))

# ---------------- Preprocessing ----------------
    st.subheader("🧹 Preprocessing")

    fill_cols = st.multiselect("Columns", df.columns)
    fill_method = st.selectbox("Method", ["Mean","Median","Mode","Forward Fill","Backward Fill"])

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

# ---------------- Encoding ----------------
    cat_cols = df.select_dtypes(include="object").columns
    encode_cols = st.multiselect("Categorical Columns", cat_cols)

    if st.button("Apply Encoding"):
        for col in encode_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        st.session_state.df = df
        st.success("✅ Encoding Applied")

# ---------------- Scaling ----------------
    num_cols = df.select_dtypes(include=np.number).columns
    scale_cols = st.multiselect("Columns for Scaling", num_cols)

    scale_method = st.selectbox("Scaling Method", ["Standardization","Normalization"])

    if st.button("Apply Scaling"):
        scaler = StandardScaler() if scale_method=="Standardization" else MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        st.session_state.df = df
        st.success("✅ Scaling Applied")

# ---------------- Model Setup ----------------
    st.subheader("⚙️ Model Setup")

    target = st.selectbox("Target Column", df.columns)

    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]

# ---------------- TASK ----------------
    target_type = type_of_target(y)

    if target_type in ["binary", "multiclass"]:
        task = "Classification"
    else:
        task = "Regression"

    st.write(f"🎯 Task: {task}")

# ---------------- Feature Selection ----------------
    k = st.slider("Top K Features",1,X.shape[1],min(5,X.shape[1]))

    selector = SelectKBest(
        f_classif if task=="Classification" else f_regression,
        k=k
    )

    X_new = selector.fit_transform(X,y)
    selected_features = X.columns[selector.get_support()]
    X = pd.DataFrame(X_new,columns=selected_features)

# ---------------- Split ----------------
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# ---------------- Models ----------------
    st.subheader("🏆 Model Leaderboard")

    results=[]
    best_model=None
    best_model_name=None

# ---------------- Classification ----------------
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

        with st.spinner("🤖 Training Models..."):
            progress=st.progress(0)
            total=len(models)

            for i,(name,model) in enumerate(models.items()):

                model.fit(X_train,y_train)
                preds=model.predict(X_test)

                acc=accuracy_score(y_test,preds)
                cv=cross_val_score(model,X,y,cv=5).mean()

                results.append([name,acc,cv])

                if acc>best_score:
                    best_score=acc
                    best_model=model
                    best_model_name=name

                progress.progress((i+1)/total)
                time.sleep(0.2)

        st.success("✅ Training Completed")

        res=pd.DataFrame(results,columns=["Model","Accuracy","CV Score"])
        st.dataframe(res)
        st.plotly_chart(px.bar(res,x="Model",y="Accuracy"))

# ---------------- Regression ----------------
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

        with st.spinner("🤖 Training Models..."):
            progress=st.progress(0)
            total=len(models)

            for i,(name,model) in enumerate(models.items()):

                model.fit(X_train,y_train)
                preds=model.predict(X_test)

                rmse=np.sqrt(mean_squared_error(y_test,preds))
                cv=cross_val_score(model,X,y,cv=5,scoring="neg_mean_squared_error").mean()

                results.append([name,rmse,cv])

                if rmse<best_score:
                    best_score=rmse
                    best_model=model
                    best_model_name=name

                progress.progress((i+1)/total)
                time.sleep(0.2)

        st.success("✅ Training Completed")

        res=pd.DataFrame(results,columns=["Model","RMSE","CV Score"])
        st.dataframe(res)
        st.plotly_chart(px.bar(res,x="Model",y="RMSE"))

# ---------------- Best Model ----------------
    st.subheader("🥇 Best Model")
    st.success(f"Best Model Selected: {best_model_name}")

    preds = best_model.predict(X_test)

# ---------------- Evaluation ----------------
    st.subheader("📊 Model Evaluation")

    if task=="Classification":
        st.write("Accuracy:",accuracy_score(y_test,preds))
        st.write("Precision:",precision_score(y_test,preds,average="weighted",zero_division=0))
        st.write("Recall:",recall_score(y_test,preds,average="weighted",zero_division=0))
        st.write("F1 Score:",f1_score(y_test,preds,average="weighted",zero_division=0))
        st.plotly_chart(px.imshow(confusion_matrix(y_test,preds),text_auto=True))

    else:
        st.write("RMSE:",np.sqrt(mean_squared_error(y_test,preds)))
        st.write("MAE:",mean_absolute_error(y_test,preds))
        st.write("R2:",r2_score(y_test,preds))

# ---------------- Prediction ----------------
    st.subheader("🔮 Prediction")

    user_input={}
    for col in X.columns:
        user_input[col]=st.number_input(col, value=float(X[col].mean()))

    input_df=pd.DataFrame([user_input])

    if st.button("Predict"):
        with st.spinner("🔮 Predicting..."):
            time.sleep(1)
            pred=best_model.predict(input_df)
        st.success(f"Prediction: {pred[0]}")

else:
    st.markdown("""
    <div class="upload-box">
    📂 Upload dataset to start AutoML 🚀
    </div>
    """, unsafe_allow_html=True)
