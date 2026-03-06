import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, cross_val_score

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

st.set_page_config(page_title="Fast AutoML Studio", layout="wide")

st.title("🚀 Fast AutoML Studio")

# ---------------- Upload Dataset ----------------

st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.success("Dataset Loaded")

# ---------------- Preview ----------------

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# ---------------- Info ----------------

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

    fill_method = st.selectbox(
        "Fill Missing Values",
        ["None","Mean","Median","Mode"]
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

# ---------------- Encoding ----------------

    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:

        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ---------------- Scaling ----------------

    scale_method = st.selectbox(
        "Scaling",
        ["None","Standardization","Normalization"]
    )

    num_cols = df.select_dtypes(include=np.number).columns

    if scale_method == "Standardization":

        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    elif scale_method == "Normalization":

        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

# ---------------- Model Setup ----------------

    st.subheader("Model Setup")

    target = st.selectbox("Target Column", df.columns)

    task = st.radio("Task Type", ["Classification","Regression"])

# ---------------- Feature Selection ----------------

    st.subheader("Feature Selection")

    X = df.drop(columns=[target])
    X = X.select_dtypes(include=np.number)

    y = df[target]

    data = pd.concat([X,y],axis=1).dropna()

    X = data.drop(columns=[target])
    y = data[target]

    k = st.slider("Top K Features",1,X.shape[1],min(3,X.shape[1]))

    selector = SelectKBest(
        f_classif if task=="Classification" else f_regression,
        k=k
    )

    X_new = selector.fit_transform(X,y)

    selected_features = X.columns[selector.get_support()]

    X = pd.DataFrame(X_new,columns=selected_features)

    st.write("Selected Features:",list(selected_features))

# ---------------- Train Test ----------------

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

# ---------------- Models ----------------

    st.subheader("Model Leaderboard")

    results = []
    best_model = None
    best_model_name = None
    best_score = None

# ---------------- Classification ----------------

    if task=="Classification":

        models = {

            "Logistic Regression": LogisticRegression(max_iter=500),

            "Random Forest": RandomForestClassifier(n_estimators=50),

            "Extra Trees": ExtraTreesClassifier(n_estimators=50),

            "Gradient Boosting": GradientBoostingClassifier(),

            "Decision Tree": DecisionTreeClassifier(),

            "KNN": KNeighborsClassifier(),

            "Naive Bayes": GaussianNB()

        }

        for name,model in models.items():

            model.fit(X_train,y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test,preds)

            cv = cross_val_score(model,X,y,cv=3).mean()

            results.append([name,acc,cv])

            if best_score is None or acc > best_score:

                best_score = acc
                best_model = model
                best_model_name = name

        res = pd.DataFrame(results,columns=["Model","Accuracy","CV Score"])

        st.dataframe(res)

        st.plotly_chart(px.bar(res,x="Model",y="Accuracy"))

# ---------------- Regression ----------------

    else:

        models = {

            "Linear Regression": LinearRegression(),

            "Ridge": Ridge(),

            "Lasso": Lasso(),

            "Random Forest": RandomForestRegressor(n_estimators=50),

            "Extra Trees": ExtraTreesRegressor(n_estimators=50),

            "Gradient Boosting": GradientBoostingRegressor(),

            "Decision Tree": DecisionTreeRegressor(),

            "KNN": KNeighborsRegressor()

        }

        for name,model in models.items():

            model.fit(X_train,y_train)

            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test,preds))

            cv = cross_val_score(model,X,y,cv=3,
                                 scoring="neg_mean_squared_error").mean()

            results.append([name,rmse,cv])

            if best_score is None or rmse < best_score:

                best_score = rmse
                best_model = model
                best_model_name = name

        res = pd.DataFrame(results,columns=["Model","RMSE","CV Score"])

        st.dataframe(res)

        st.plotly_chart(px.bar(res,x="Model",y="RMSE"))

# ---------------- Best Model ----------------

    st.subheader("🏆 Best Model")

    st.success(best_model_name)

    preds = best_model.predict(X_test)

# ---------------- Evaluation ----------------

    if task=="Classification":

        st.write("Accuracy:",accuracy_score(y_test,preds))
        st.write("Precision:",precision_score(y_test,preds,average="weighted"))
        st.write("Recall:",recall_score(y_test,preds,average="weighted"))
        st.write("F1:",f1_score(y_test,preds,average="weighted"))

        cm = confusion_matrix(y_test,preds)

        st.plotly_chart(px.imshow(cm,text_auto=True))

    else:

        st.write("RMSE:",np.sqrt(mean_squared_error(y_test,preds)))
        st.write("MAE:",mean_absolute_error(y_test,preds))
        st.write("R2:",r2_score(y_test,preds))

# ---------------- Feature Importance ----------------

    if hasattr(best_model,"feature_importances_"):

        st.subheader("Feature Importance")

        importance = pd.DataFrame({
            "Feature":X.columns,
            "Importance":best_model.feature_importances_
        })

        st.plotly_chart(px.bar(importance,x="Feature",y="Importance"))

# ---------------- Download Model ----------------

    st.subheader("Download Model")

    model_bytes = pickle.dumps(best_model)

    st.download_button(
        "Download Model",
        model_bytes,
        "best_model.pkl"
    )

# ---------------- Prediction ----------------

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

        st.success(pred[0])

else:

    st.info("Upload dataset to start AutoML")
