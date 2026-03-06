import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Regression models
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

st.sidebar.header("📂 Upload Dataset")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("Dataset Loaded Successfully")

    # ---------------- DATA PREVIEW ----------------

    st.subheader("📊 Dataset Preview")

    st.dataframe(df.head())

    # ---------------- DATA INFO ----------------

    st.subheader("📋 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

    # ---------------- EDA ----------------

    st.subheader("📈 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Distribution Plot Column",
            numeric_cols
        )

        fig = px.histogram(df, x=selected_col)

        st.plotly_chart(fig)

    # ---------------- CORRELATION ----------------

    if len(numeric_cols) > 1:

        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig = px.imshow(corr, text_auto=True)

        st.plotly_chart(fig)

    # ---------------- PREPROCESSING ----------------

    st.subheader("🧹 Preprocessing")

    st.write("### Missing Value Count")

    st.write(df.isnull().sum())

    # Missing fill

    fill_cols = st.multiselect("Columns for Missing Fill", df.columns)

    fill_method = st.selectbox(
        "Fill Method",
        ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill"]
    )

    if st.button("Apply Missing Fill"):

        for col in fill_cols:

            if fill_method == "Mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())

            elif fill_method == "Median":
                if pd.api.types.is_numeric_dtype(df[col]):
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

    st.write("### Encoding")

    cat_cols = df.select_dtypes(include="object").columns

    encode_cols = st.multiselect("Categorical Columns", cat_cols)

    if st.button("Apply Encoding"):

        for col in encode_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        st.session_state.df = df

        st.success("Encoding applied")

    # ---------------- SCALING ----------------

    st.write("### Scaling")

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

        st.success("Scaling applied")

    # ---------------- TARGET ----------------

    st.subheader("🎯 Model Setup")

    target = st.selectbox("Target Column", df.columns)

    task = st.radio(
        "Task Type",
        ["Classification", "Regression"]
    )

    X = df.drop(columns=[target])
    y = df[target]

    # ---------------- FEATURE SELECTION ----------------

    st.subheader("🎯 Feature Selection")

    k = st.slider("Top K Features", 1, X.shape[1], min(5, X.shape[1]))

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
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Extra Trees": ExtraTreesClassifier()

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
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "Extra Trees": ExtraTreesRegressor()

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

    # ---------------- DOWNLOAD MODEL ----------------

    st.subheader("💾 Download Model")

    model_bytes = pickle.dumps(best_model)

    st.download_button(
        label="Download Trained Model",
        data=model_bytes,
        file_name="trained_model.pkl"
    )

    # ---------------- PREDICTION ----------------

    st.subheader("🔮 Prediction")

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

    st.info("Upload a dataset to start AutoML")
