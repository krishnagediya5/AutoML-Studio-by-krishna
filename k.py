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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc
)

st.set_page_config(page_title="AutoML Studio", layout="wide")

# ---------------- TITLE + HERO SECTION ----------------

st.title("🚀 AutoML Pro Studio")

st.markdown("""
### 🔥 Build Machine Learning Models in Seconds — No Coding Required

AutoML Pro Studio is an intelligent platform that allows you to upload your dataset, preprocess data, train multiple machine learning models, and generate accurate predictions — all in one place.

👉 No coding. No complexity. Just powerful AI.
""")

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ---------------- Upload Dataset ----------------

st.sidebar.header("📂 Upload Dataset")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("✅ Dataset Loaded Successfully")

# ---------------- Dataset Preview ----------------

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

# ---------------- Dataset Info ----------------

    col1, col2 = st.columns(2)

    with col1:
        st.write("📐 Shape:", df.shape)

    with col2:
        st.write("❗ Missing Values")
        st.write(df.isnull().sum())

# ---------------- EDA ----------------

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        col = st.selectbox("📈 Distribution Column", numeric_cols)

        fig = px.histogram(df, x=col)

        st.plotly_chart(fig)

# ---------------- Preprocessing ----------------

    st.subheader("🧹 Preprocessing")

    fill_cols = st.multiselect("Columns for Missing Fill", df.columns)

    fill_method = st.selectbox(
        "Fill Method",
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

    scale_method = st.selectbox(
        "Scaling Method",
        ["Standardization","Normalization"]
    )

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

# ---------------- AUTO TASK DETECTION ----------------

    target_type = type_of_target(y)

    if target_type in ["binary", "multiclass"]:
        task = "Classification"
    else:
        task = "Regression"

    st.write("🎯 Detected Task:", task)

    if task == "Classification" and type_of_target(y) == "continuous":
        st.error("❌ Continuous target cannot be used for classification.")
        st.stop()

# ---------------- Feature Selection ----------------

    st.subheader("🎯 Feature Selection")

    k = st.slider("Top K Features",1,X.shape[1],min(5,X.shape[1]))

    selector = SelectKBest(
        f_classif if task=="Classification" else f_regression,
        k=k
    )

    X_new = selector.fit_transform(X,y)

    selected_features = X.columns[selector.get_support()]

    X = pd.DataFrame(X_new,columns=selected_features)

    st.write("Selected Features:",list(selected_features))

# ---------------- Train Test Split ----------------

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

# ---------------- Models ----------------

    st.subheader("🏆 Model Leaderboard")

    results = []

    best_model = None
    best_model_name = None

# ---------------- Classification ----------------

    if task=="Classification":

        best_score = 0

        models = {

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

            preds = model.predict(X_test)

            acc = accuracy_score(y_test,preds)

            cv = cross_val_score(model,X,y,cv=5).mean()

            results.append([name,acc,cv])

            if acc > best_score:
                best_score = acc
                best_model = model
                best_model_name = name

        res = pd.DataFrame(results,columns=["Model","Accuracy","CV Score"])

        st.dataframe(res)

        fig = px.bar(res,x="Model",y="Accuracy",title="Model Comparison")

        st.plotly_chart(fig)

# ---------------- Regression ----------------

    else:

        best_score = float("inf")

        models = {

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

            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test,preds))

            cv = cross_val_score(model,X,y,cv=5,
                                 scoring="neg_mean_squared_error").mean()

            results.append([name,rmse,cv])

            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_model_name = name

        res = pd.DataFrame(results,columns=["Model","RMSE","CV Score"])

        st.dataframe(res)

        fig = px.bar(res,x="Model",y="RMSE",title="Model Comparison")

        st.plotly_chart(fig)

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

        cm = confusion_matrix(y_test,preds)

        fig = px.imshow(cm,text_auto=True)

        st.plotly_chart(fig)

        if hasattr(best_model,"predict_proba") and len(np.unique(y_test))==2:

            probs = best_model.predict_proba(X_test)[:,1]

            fpr,tpr,_ = roc_curve(y_test,probs)

            roc_auc = auc(fpr,tpr)

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=fpr,y=tpr,name="ROC Curve"))

            fig.update_layout(
                title=f"ROC Curve (AUC={roc_auc:.2f})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )

            st.plotly_chart(fig)

    else:

        st.write("RMSE:",np.sqrt(mean_squared_error(y_test,preds)))
        st.write("MAE:",mean_absolute_error(y_test,preds))
        st.write("R2 Score:",r2_score(y_test,preds))

# ---------------- Feature Importance ----------------

    st.subheader("📊 Feature Importance")

    if hasattr(best_model,"feature_importances_"):

        importance = pd.DataFrame({
            "Feature":X.columns,
            "Importance":best_model.feature_importances_
        })

        importance = importance.sort_values(by="Importance",ascending=False)

        fig = px.bar(importance,x="Feature",y="Importance")

        st.plotly_chart(fig)

    else:

        st.info("Feature importance not available")

# ---------------- Download Model ----------------

    st.subheader("📥 Download Model")

    model_bytes = pickle.dumps(best_model)

    st.download_button(
        label="Download Trained Model",
        data=model_bytes,
        file_name="best_model.pkl"
    )

# ---------------- Prediction ----------------

    st.subheader("🔮 Prediction")

    user_input = {}

    for col in X.columns:

        user_input[col] = st.number_input(
            f"Enter {col}",
            value=float(X[col].mean())
        )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):

        pred = best_model.predict(input_df)

        st.success(f"Prediction: {pred[0]}")

else:

    st.markdown("""
    <div style="background-color:#E8F0FE;padding:20px;border-radius:10px;text-align:center">
    <h4>📂 Upload your dataset to begin building powerful ML models instantly 🚀</h4>
    </div>
    """, unsafe_allow_html=True)
