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

st.set_page_config(page_title="AutoML Pro", layout="wide")
st.title("🚀 AutoML Pro Studio")

# ---------------- Upload ----------------
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)
    st.success("Dataset Loaded Successfully")

    st.subheader("Preview")
    st.dataframe(df.head())

# ---------------- Target ----------------
    target = st.selectbox("Select Target Column", df.columns)

    # Drop missing target rows
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

# ---------------- Task ----------------
    task_mode = st.radio("Task Type", ["Auto", "Classification", "Regression"])

    detected = type_of_target(y)

    if task_mode == "Auto":
        task = "Classification" if detected in ["binary","multiclass"] else "Regression"
    else:
        task = task_mode

    st.write("Detected Task:", task)

# 🚨 SAFETY GUARD (NO MORE ERRORS)
    if task == "Classification" and type_of_target(y) == "continuous":
        st.error("❌ Continuous target cannot be used for classification. Switch to Regression.")
        st.stop()

# ---------------- Encoding ----------------
    cat_cols = X.select_dtypes(include="object").columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# ---------------- Scaling ----------------
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ---------------- Feature Selection ----------------
    k = st.slider("Top Features", 1, X.shape[1], min(5, X.shape[1]))

    selector = SelectKBest(
        f_classif if task=="Classification" else f_regression,
        k=k
    )

    X_new = selector.fit_transform(X, y)
    selected = X.columns[selector.get_support()]
    X = pd.DataFrame(X_new, columns=selected)

    st.write("Selected Features:", list(selected))

# ---------------- Split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# ---------------- Models ----------------
    st.subheader("Model Leaderboard")

    results = []
    best_model = None
    best_name = None

# ---------------- Classification ----------------
    if task == "Classification":

        best_score = 0

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True),
            "Naive Bayes": GaussianNB()
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                cv = cross_val_score(model, X, y, cv=5).mean()

                results.append([name, acc, cv])

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_name = name

            except:
                continue

        res = pd.DataFrame(results, columns=["Model","Accuracy","CV"])
        st.dataframe(res)
        st.plotly_chart(px.bar(res, x="Model", y="Accuracy"))

# ---------------- Regression ----------------
    else:

        best_score = float("inf")

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest": RandomForestRegressor(),
            "Extra Trees": ExtraTreesRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR()
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, preds))
                cv = cross_val_score(model, X, y, cv=5,
                                     scoring="neg_mean_squared_error").mean()

                results.append([name, rmse, cv])

                if rmse < best_score:
                    best_score = rmse
                    best_model = model
                    best_name = name

            except:
                continue

        res = pd.DataFrame(results, columns=["Model","RMSE","CV"])
        st.dataframe(res)
        st.plotly_chart(px.bar(res, x="Model", y="RMSE"))

# ---------------- Best ----------------
    st.subheader("🥇 Best Model")
    st.success(best_name)

    preds = best_model.predict(X_test)

# ---------------- Evaluation ----------------
    if task == "Classification":

        st.write("Accuracy:", accuracy_score(y_test, preds))
        st.write("F1:", f1_score(y_test, preds, average="weighted"))

        cm = confusion_matrix(y_test, preds)
        st.plotly_chart(px.imshow(cm, text_auto=True))

    else:

        st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
        st.write("R2:", r2_score(y_test, preds))

# ---------------- Download ----------------
    st.subheader("Download Model")

    st.download_button(
        "Download Model",
        pickle.dumps(best_model),
        "model.pkl"
    )

# ---------------- Predict ----------------
    st.subheader("Prediction")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(col, value=float(X[col].mean()))

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        pred = best_model.predict(input_df)
        st.success(f"Prediction: {pred[0]}")

else:
    st.info("Upload a dataset to begin")
