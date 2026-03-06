import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, cross_val_score

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

st.set_page_config(page_title="AutoML Pro Studio", layout="wide")

st.title("🚀 AutoML Pro Studio")

# ---------------- Dataset Upload ----------------

st.sidebar.header("Upload Datasets")

file1 = st.sidebar.file_uploader("Upload Dataset 1", type=["csv"])
file2 = st.sidebar.file_uploader("Upload Dataset 2 (optional)", type=["csv"])

# ---------------- Dataset Comparison ----------------

if file1 and file2:

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    st.header("📊 Dataset Comparison")

    shape_df = pd.DataFrame({
        "Dataset":["Dataset 1","Dataset 2"],
        "Rows":[df1.shape[0],df2.shape[0]],
        "Columns":[df1.shape[1],df2.shape[1]]
    })

    st.subheader("Shape Comparison")
    st.dataframe(shape_df)

    # Missing comparison
    missing1 = df1.isnull().sum().sum()
    missing2 = df2.isnull().sum().sum()

    missing_df = pd.DataFrame({
        "Dataset":["Dataset 1","Dataset 2"],
        "Missing Values":[missing1,missing2]
    })

    st.subheader("Missing Values Comparison")
    st.bar_chart(missing_df.set_index("Dataset"))

    # Feature distribution comparison
    common_cols = list(set(df1.columns).intersection(set(df2.columns)))

    if common_cols:

        col = st.selectbox("Select Feature to Compare", common_cols)

        fig1 = px.histogram(df1, x=col, title="Dataset 1 Distribution")
        fig2 = px.histogram(df2, x=col, title="Dataset 2 Distribution")

        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

# ---------------- Main AutoML Dataset ----------------

if file1:

    df = pd.read_csv(file1)

    st.header("📊 AutoML Training Dataset")

    st.dataframe(df.head())

    col1,col2 = st.columns(2)

    with col1:
        st.write("Shape:",df.shape)

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

# ---------------- EDA ----------------

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols)>0:

        col = st.selectbox("Distribution Column",numeric_cols)

        fig = px.histogram(df,x=col)

        st.plotly_chart(fig)

# ---------------- Preprocessing ----------------

    st.subheader("Preprocessing")

    fill_cols = st.multiselect("Columns for Missing Fill",df.columns)

    fill_method = st.selectbox(
        "Fill Method",
        ["Mean","Median","Mode","Forward Fill","Backward Fill"]
    )

    if st.button("Apply Missing Fill"):

        for col in fill_cols:

            if fill_method=="Mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())

            elif fill_method=="Median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())

            elif fill_method=="Mode":
                df[col] = df[col].fillna(df[col].mode()[0])

            elif fill_method=="Forward Fill":
                df[col] = df[col].ffill()

            elif fill_method=="Backward Fill":
                df[col] = df[col].bfill()

        st.success("Missing values handled")

# ---------------- Encoding ----------------

    cat_cols = df.select_dtypes(include="object").columns

    encode_cols = st.multiselect("Categorical Columns",cat_cols)

    if st.button("Apply Encoding"):

        for col in encode_cols:

            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        st.success("Encoding applied")

# ---------------- Scaling ----------------

    num_cols = df.select_dtypes(include=np.number).columns

    scale_cols = st.multiselect("Scaling Columns",num_cols)

    scale_method = st.selectbox(
        "Scaling Method",
        ["Standardization","Normalization"]
    )

    if st.button("Apply Scaling"):

        scaler = StandardScaler() if scale_method=="Standardization" else MinMaxScaler()

        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        st.success("Scaling applied")

# ---------------- Model Setup ----------------

    st.subheader("Model Setup")

    target = st.selectbox("Target Column",df.columns)

    task = st.radio("Task Type",["Classification","Regression"])

    X = df.drop(columns=[target])
    y = df[target]

# ---------------- Feature Selection ----------------

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

# ---------------- Model Training ----------------

    st.subheader("Model Leaderboard")

    results = []
    best_score = -999
    best_model = None
    best_model_name = None

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

        for name,model in models.items():

            model.fit(X_train,y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test,preds)

            cv = cross_val_score(model,X,y,cv=5).mean()

            results.append([name,acc,cv])

            if acc>best_score:
                best_score=acc
                best_model=model
                best_model_name=name

        res=pd.DataFrame(results,columns=["Model","Accuracy","CV Score"])

        st.dataframe(res)

        fig=px.bar(res,x="Model",y="Accuracy",title="Model Comparison")

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

        for name,model in models.items():

            model.fit(X_train,y_train)

            preds=model.predict(X_test)

            rmse=np.sqrt(mean_squared_error(y_test,preds))

            cv=cross_val_score(model,X,y,cv=5,
                               scoring="neg_mean_squared_error").mean()

            results.append([name,rmse,cv])

            if best_model is None or rmse<best_score:
                best_score=rmse
                best_model=model
                best_model_name=name

        res=pd.DataFrame(results,columns=["Model","RMSE","CV Score"])

        st.dataframe(res)

        fig=px.bar(res,x="Model",y="RMSE",title="Model Comparison")

        st.plotly_chart(fig)

# ---------------- Best Model ----------------

    st.subheader("🏆 Best Model")

    st.success(f"Best Model Selected: {best_model_name}")

    preds=best_model.predict(X_test)

# ---------------- Evaluation ----------------

    st.subheader("Model Evaluation")

    if task=="Classification":

        st.write("Accuracy:",accuracy_score(y_test,preds))
        st.write("Precision:",precision_score(y_test,preds,average="weighted"))
        st.write("Recall:",recall_score(y_test,preds,average="weighted"))
        st.write("F1:",f1_score(y_test,preds,average="weighted"))

        cm=confusion_matrix(y_test,preds)

        st.plotly_chart(px.imshow(cm,text_auto=True))

    else:

        st.write("RMSE:",np.sqrt(mean_squared_error(y_test,preds)))
        st.write("MAE:",mean_absolute_error(y_test,preds))
        st.write("R2:",r2_score(y_test,preds))

# ---------------- Feature Importance ----------------

    if hasattr(best_model,"feature_importances_"):

        importance=pd.DataFrame({
            "Feature":X.columns,
            "Importance":best_model.feature_importances_
        })

        importance=importance.sort_values(by="Importance",ascending=False)

        st.plotly_chart(px.bar(importance,x="Feature",y="Importance"))

# ---------------- Download Model ----------------

    model_bytes=pickle.dumps(best_model)

    st.download_button(
        label="Download Trained Model",
        data=model_bytes,
        file_name="best_model.pkl"
    )

# ---------------- Prediction ----------------

    st.subheader("Prediction")

    user_input={}

    for col in X.columns:

        user_input[col]=st.number_input(
            f"Enter {col}",
            value=float(X[col].mean())
        )

    input_df=pd.DataFrame([user_input])

    if st.button("Predict"):

        pred=best_model.predict(input_df)

        st.success(f"Prediction: {pred[0]}")

else:

    st.info("Upload dataset to start AutoML")
