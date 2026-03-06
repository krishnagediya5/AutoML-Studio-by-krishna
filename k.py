import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc
)

st.set_page_config(page_title="AutoML Pro", layout="wide")

st.title("🚀 AutoML Pro Studio")


# ---------------- CACHE DATA ----------------

@st.cache_data
def load_data(file):
    return pd.read_csv(file)


# ---------------- Upload Dataset ----------------

st.sidebar.header("Upload Dataset")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    if "df" not in st.session_state:
        st.session_state.df = load_data(file)

    df = st.session_state.df

    st.success("Dataset Loaded Successfully")

# ---------------- Dataset Preview ----------------

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# ---------------- Dataset Info ----------------

    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

# ---------------- EDA ----------------

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        col = st.selectbox("Distribution Column", numeric_cols)

        fig = px.histogram(df, x=col)

        st.plotly_chart(fig)

# ---------------- Preprocessing ----------------

    st.subheader("Preprocessing")

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
        st.success("Missing Values Handled")

# ---------------- Encoding ----------------

    cat_cols = df.select_dtypes(include="object").columns

    encode_cols = st.multiselect("Categorical Columns", cat_cols)

    if st.button("Apply Encoding"):

        for col in encode_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        st.session_state.df = df
        st.success("Encoding Applied")

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

        st.success("Scaling Applied")

# ---------------- Model Setup ----------------

    st.subheader("Model Setup")

    target = st.selectbox("Target Column", df.columns)

    task = st.radio("Task Type", ["Classification","Regression"])

    X = df.drop(columns=[target])
    y = df[target]

# ---------------- Feature Selection ----------------

    st.subheader("Feature Selection")

    k = st.slider("Top K Features",1,X.shape[1],min(5,X.shape[1]))

    selector = SelectKBest(
        f_classif if task=="Classification" else f_regression,
        k=k
    )

    X_new = selector.fit_transform(X,y)

    selected_features = X.columns[selector.get_support()]

    X = pd.DataFrame(X_new,columns=selected_features)

    st.write("Selected Features:",list(selected_features))

# ---------------- Train Button ----------------

    if st.button("🚀 Train Models"):

        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=0.2,random_state=42
        )

        st.subheader("Model Leaderboard")

        results = []

        progress = st.progress(0)

# ---------------- Classification ----------------

        if task=="Classification":

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

            best_score = 0

            for i,(name,model) in enumerate(models.items()):

                model.fit(X_train,y_train)

                preds = model.predict(X_test)

                acc = accuracy_score(y_test,preds)

                cv = cross_val_score(model,X,y,cv=5).mean()

                results.append([name,acc,cv])

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_model_name = name

                progress.progress((i+1)/len(models))

            res = pd.DataFrame(results,columns=["Model","Accuracy","CV Score"])

            st.dataframe(res)

            fig = px.bar(res,x="Model",y="Accuracy")

            st.plotly_chart(fig)

# ---------------- Regression ----------------

        else:

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

            best_score = 999999

            for i,(name,model) in enumerate(models.items()):

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

                progress.progress((i+1)/len(models))

            res = pd.DataFrame(results,columns=["Model","RMSE","CV Score"])

            st.dataframe(res)

            fig = px.bar(res,x="Model",y="RMSE")

            st.plotly_chart(fig)

# ---------------- Best Model ----------------

        st.subheader("🥇 Best Model")

        st.success(f"Best Model Selected: {best_model_name}")

# ---------------- Download ----------------

        model_bytes = pickle.dumps(best_model)

        st.download_button(
            label="Download Trained Model",
            data=model_bytes,
            file_name="best_model.pkl"
        )

else:

    st.info("Upload dataset to start AutoML")
