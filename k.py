import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Optional explainability
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Regression models
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

st.set_page_config(page_title="AutoML Pro Studio", layout="wide")

st.title("🚀 AutoML Pro Studio (Advanced)")

# ================= Upload =================

st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("Dataset Loaded Successfully")

# ================= Preview =================

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

# ================= Correlation Heatmap =================

    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig)

# ================= EDA =================

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        col = st.selectbox("Distribution Column", numeric_cols)

        fig = px.histogram(df, x=col)

        st.plotly_chart(fig)

# ================= Preprocessing =================

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

# ================= Encoding =================

    cat_cols = df.select_dtypes(include="object").columns

    encode_cols = st.multiselect("Categorical Columns", cat_cols)

    if st.button("Apply Encoding"):

        for col in encode_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        st.session_state.df = df
        st.success("Encoding Applied")

# ================= Scaling =================

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

# ================= Model Setup =================

    st.subheader("Model Setup")

    target = st.selectbox("Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

# ================= Auto Task Detection =================

    if y.dtype == "object" or len(y.unique()) <= 10:
        task = "Classification"
    else:
        task = "Regression"

    st.write("Detected Task:", task)

# ================= Feature Selection =================

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

# ================= Train Test Split =================

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

# ================= Models =================

    st.subheader("Model Leaderboard")

    results = []

    best_model = None
    best_model_name = None

    progress = st.progress(0)

# ================= Classification =================

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

# ================= Regression =================

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

        for i,(name,model) in enumerate(models.items()):

            model.fit(X_train,y_train)

            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test,preds))

            cv = cross_val_score(
                model,X,y,cv=5,
                scoring="neg_mean_squared_error"
            ).mean()

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

# ================= Best Model =================

    st.subheader("🏆 Best Model")

    st.success(f"Best Model Selected: {best_model_name}")

    preds = best_model.predict(X_test)

# ================= Evaluation =================

    st.subheader("Model Evaluation")

    if task=="Classification":

        st.write("Accuracy:",accuracy_score(y_test,preds))
        st.write("Precision:",precision_score(y_test,preds,average="weighted",zero_division=0))
        st.write("Recall:",recall_score(y_test,preds,average="weighted",zero_division=0))
        st.write("F1 Score:",f1_score(y_test,preds,average="weighted",zero_division=0))

        cm = confusion_matrix(y_test,preds)

        fig = px.imshow(cm,text_auto=True)

        st.plotly_chart(fig)

    else:

        st.write("RMSE:",np.sqrt(mean_squared_error(y_test,preds)))
        st.write("MAE:",mean_absolute_error(y_test,preds))
        st.write("R2 Score:",r2_score(y_test,preds))

# ================= Feature Importance =================

    st.subheader("Feature Importance")

    importance = None

    if hasattr(best_model,"feature_importances_"):
        importance = best_model.feature_importances_

    elif hasattr(best_model,"coef_"):
        importance = np.abs(best_model.coef_)

    if importance is not None:

        imp_df = pd.DataFrame({
            "Feature":X.columns,
            "Importance":importance
        })

        imp_df = imp_df.sort_values(by="Importance",ascending=False)

        fig = px.bar(imp_df,x="Feature",y="Importance")

        st.plotly_chart(fig)

# ================= SHAP =================

    if SHAP_AVAILABLE:

        st.subheader("Model Explainability (SHAP)")

        try:

            explainer = shap.Explainer(best_model, X_train)

            shap_values = explainer(X_test)

            fig = shap.plots.bar(shap_values, show=False)

            st.pyplot(fig)

        except:
            st.info("SHAP not supported for this model")

# ================= Hyperparameter Tuning =================

    if st.button("Run Hyperparameter Tuning"):

        param_grid = {
            "n_estimators":[100,200,300],
            "max_depth":[None,5,10]
        }

        tuner = RandomizedSearchCV(
            RandomForestRegressor() if task=="Regression" else RandomForestClassifier(),
            param_grid,
            n_iter=5,
            cv=3,
            n_jobs=-1
        )

        tuner.fit(X_train,y_train)

        best_model = tuner.best_estimator_

        st.success("Hyperparameter tuning completed")

# ================= Download Model =================

    st.subheader("Download Model")

    model_bytes = pickle.dumps(best_model)

    st.download_button(
        label="Download Trained Model",
        data=model_bytes,
        file_name="best_model.pkl"
    )

# ================= Prediction =================

    st.subheader("Prediction")

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

    st.info("Upload dataset to start AutoML")
