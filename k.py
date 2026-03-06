import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import accuracy_score, mean_squared_error

st.set_page_config(page_title="AutoML Pro", layout="wide")

st.title("🚀 AutoML Pro Studio")
st.write("Advanced End-to-End Machine Learning Platform")

# Upload Dataset
st.sidebar.header("📂 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.success("Dataset Loaded Successfully")

    # ---------------- DATA PREVIEW ----------------
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- BASIC INFO ----------------
    st.subheader("📋 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

    # ---------------- EDA ----------------
    st.subheader("📈 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        selected_col = st.selectbox("Select Column for Distribution", numeric_cols)

        fig = px.histogram(df, x=selected_col)

        st.plotly_chart(fig)

    # ---------------- CORRELATION ----------------
    st.subheader("🔥 Correlation Heatmap")

    if len(numeric_cols) > 1:

        corr = df[numeric_cols].corr()

        fig = px.imshow(corr, text_auto=True)

        st.plotly_chart(fig)

    # ---------------- PREPROCESSING ----------------
    st.subheader("🧹 Preprocessing")

    # Fill Missing
    fill_method = st.selectbox(
        "Fill Missing Values",
        ["None", "Mean", "Median", "Mode"]
    )

    if fill_method != "None":

        for col in df.columns:

            if df[col].isnull().sum() > 0:

                if fill_method == "Mean" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())

                elif fill_method == "Median" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())

                elif fill_method == "Mode":
                    df[col] = df[col].fillna(df[col].mode()[0])

    # Encoding
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Scaling
    scaler = StandardScaler()

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ---------------- TARGET ----------------
    st.subheader("🎯 Model Setup")

    target = st.selectbox("Select Target Column", df.columns)

    task = st.radio(
        "Task Type",
        ["Classification", "Regression"]
    )

    X = df.drop(columns=[target])
    y = df[target]

    # ---------------- FEATURE SELECTION ----------------
    k = st.slider("Select Top K Features", 1, X.shape[1], min(5, X.shape[1]))

    selector = SelectKBest(
        f_classif if task == "Classification" else f_regression,
        k=k
    )

    X_new = selector.fit_transform(X, y)

    selected_features = X.columns[selector.get_support()]

    X = pd.DataFrame(X_new, columns=selected_features)

    st.write("Selected Features:", list(selected_features))

    # ---------------- TRAIN TEST ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- MODELS ----------------
    st.subheader("🤖 Model Leaderboard")

    results = []

    if task == "Classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            results.append([name, acc])

        res = pd.DataFrame(results, columns=["Model", "Accuracy"])

        st.dataframe(res)

        best_model_name = res.sort_values(
            "Accuracy", ascending=False
        ).iloc[0]["Model"]

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))

            results.append([name, rmse])

        res = pd.DataFrame(results, columns=["Model", "RMSE"])

        st.dataframe(res)

        best_model_name = res.sort_values("RMSE").iloc[0]["Model"]

    best_model = models[best_model_name]

    best_model.fit(X_train, y_train)

    st.success(f"🏆 Best Model: {best_model_name}")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("📊 Feature Importance")

    if hasattr(best_model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_
        })

        fig = px.bar(importance, x="Feature", y="Importance")

        st.plotly_chart(fig)

    # ---------------- DOWNLOAD MODEL ----------------
    st.subheader("💾 Download Model")

    model_bytes = pickle.dumps(best_model)

    st.download_button(
        label="Download Trained Model",
        data=model_bytes,
        file_name="trained_model.pkl"
    )

    # ---------------- PREDICTION ----------------
    st.subheader("🔮 Make Prediction")

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

    st.info("Upload a dataset to start AutoML.")
