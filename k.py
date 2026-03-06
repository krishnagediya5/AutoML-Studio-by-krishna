import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import accuracy_score, mean_squared_error


st.set_page_config(page_title="AutoML Pro Studio", layout="wide")

st.title("🚀 AutoML Pro Studio")
st.write("Professional End-to-End Machine Learning Platform")


# ---------------- Upload Dataset ----------------

st.sidebar.header("Upload Dataset")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("Dataset Loaded Successfully")

    # ---------------- DATA PREVIEW ----------------

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # ---------------- DATA INFO ----------------

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

    # ---------------- EDA ----------------

    st.subheader("EDA")

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        col = st.selectbox("Distribution Plot Column", numeric_cols)

        fig = px.histogram(df, x=col)

        st.plotly_chart(fig)

    # ---------------- PREPROCESSING ----------------

    st.subheader("Preprocessing")

    st.write("Missing Value Count")

    st.write(df.isnull().sum())

    fill_cols = st.multiselect("Columns for Missing Fill", df.columns)

    fill_method = st.selectbox(
        "Fill Method",
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

        st.success("Missing values filled")

    # ---------------- ENCODING ----------------

    st.subheader("Encoding")

    cat_cols = df.select_dtypes(include="object").columns

    encode_cols = st.multiselect("Categorical Columns", cat_cols)

    if st.button("Apply Encoding"):

        for col in encode_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        st.session_state.df = df

        st.success("Encoding Applied")

    # ---------------- SCALING ----------------

    st.subheader("Scaling")

    num_cols = df.select_dtypes(include=np.number).columns

    scale_cols = st.multiselect("Numeric Columns", num_cols)

    scale_method = st.selectbox(
        "Scaling Method",
        ["Standardization", "Normalization"]
    )

    if st.button("Apply Scaling"):

        if scale_method == "Standardization":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        st.session_state.df = df

        st.success("Scaling Applied")

    # ---------------- MODEL SETUP ----------------

    st.subheader("Model Setup")

    target = st.selectbox("Target Column", df.columns)

    task = st.radio("Task Type", ["Classification", "Regression"])

    X = df.drop(columns=[target])
    y = df[target]

    # ---------------- FEATURE SELECTION ----------------

    st.subheader("Feature Selection")

    k = st.slider("Top K Features", 1, X.shape[1], min(5, X.shape[1]))

    selector = SelectKBest(
        f_classif if task == "Classification" else f_regression,
        k=k
    )

    X_new = selector.fit_transform(X, y)

    selected_features = X.columns[selector.get_support()]

    X = pd.DataFrame(X_new, columns=selected_features)

    st.write("Selected Features:", list(selected_features))

    # ---------------- TRAIN TEST SPLIT ----------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- HYPERPARAMETER TUNING ----------------

    st.subheader("Hyperparameter Tuning")

    tune_model = st.checkbox("Enable Hyperparameter Tuning")

    # ---------------- MODEL TRAINING ----------------

    st.subheader("Model Leaderboard")

    results = []

    if task == "Classification":

        models = {

            "Logistic Regression": (
                LogisticRegression(max_iter=1000),
                {"C":[0.01,0.1,1,10]}
            ),

            "Random Forest": (
                RandomForestClassifier(),
                {"n_estimators":[100,200,300],
                 "max_depth":[None,5,10]}
            ),

            "Decision Tree": (
                DecisionTreeClassifier(),
                {"max_depth":[None,5,10]}
            ),

            "KNN": (
                KNeighborsClassifier(),
                {"n_neighbors":[3,5,7]}
            ),

            "SVM": (
                SVC(),
                {"C":[0.1,1,10],
                 "kernel":["linear","rbf"]}
            )

        }

        for name,(model,params) in models.items():

            if tune_model:

                search = RandomizedSearchCV(
                    model,
                    params,
                    n_iter=5,
                    cv=3,
                    n_jobs=-1
                )

                search.fit(X_train,y_train)

                best = search.best_estimator_

            else:

                best = model.fit(X_train,y_train)

            preds = best.predict(X_test)

            acc = accuracy_score(y_test,preds)

            results.append([name,acc])

        res = pd.DataFrame(results,columns=["Model","Accuracy"])

        st.dataframe(res)

        best_model_name = res.sort_values(
            "Accuracy",ascending=False
        ).iloc[0]["Model"]

    else:

        models = {

            "Linear Regression": (LinearRegression(),{}),

            "Random Forest": (
                RandomForestRegressor(),
                {"n_estimators":[100,200],
                 "max_depth":[None,5,10]}
            ),

            "Decision Tree": (
                DecisionTreeRegressor(),
                {"max_depth":[None,5,10]}
            ),

            "SVR": (
                SVR(),
                {"C":[0.1,1,10]}
            )

        }

        for name,(model,params) in models.items():

            if tune_model and params:

                search = RandomizedSearchCV(
                    model,
                    params,
                    n_iter=5,
                    cv=3,
                    n_jobs=-1
                )

                search.fit(X_train,y_train)

                best = search.best_estimator_

            else:

                best = model.fit(X_train,y_train)

            preds = best.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test,preds))

            results.append([name,rmse])

        res = pd.DataFrame(results,columns=["Model","RMSE"])

        st.dataframe(res)

        best_model_name = res.sort_values("RMSE").iloc[0]["Model"]

    best_model = dict(models)[best_model_name][0]

    best_model.fit(X_train,y_train)

    st.success(f"Best Model: {best_model_name}")

    # ---------------- DOWNLOAD MODEL ----------------

    st.subheader("Download Model")

    model_bytes = pickle.dumps(best_model)

    st.download_button(
        label="Download Trained Model",
        data=model_bytes,
        file_name="trained_model.pkl"
    )

    # ---------------- PREDICTION ----------------

    st.subheader("Prediction")

    user_input = {}

    for col in X.columns:

        user_input[col] = st.number_input(
            f"Enter {col}",
            value=float(X[col].mean())
        )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):

        prediction = best_model.predict(input_df)

        if task == "Classification":
            st.success(f"Predicted Class: {prediction[0]}")
        else:
            st.success(f"Predicted Value: {prediction[0]:.4f}")

else:

    st.info("Upload dataset to start AutoML")
