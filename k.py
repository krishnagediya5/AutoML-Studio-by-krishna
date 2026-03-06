import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, learning_curve

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

st.title("🚀 Advanced AutoML Studio")

# ---------------- Upload Dataset ----------------

st.sidebar.header("Upload Dataset")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    try:
        file.seek(0)
        df = pd.read_csv(file)
    except:
        st.error("Error reading CSV file")
        st.stop()

    st.success("Dataset Loaded Successfully")

# ---------------- Dataset Preview ----------------

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# ---------------- Dataset Info ----------------

    col1,col2 = st.columns(2)

    with col1:
        st.write("Shape:",df.shape)

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

# ---------------- Dataset Quality ----------------

    st.subheader("Dataset Quality")

    missing_ratio = df.isnull().sum().sum() / df.size

    if missing_ratio < 0.05:
        quality = "Excellent"
    elif missing_ratio < 0.15:
        quality = "Good"
    else:
        quality = "Poor"

    st.metric("Dataset Quality",quality)

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

        st.success("Missing Values Handled")

# ---------------- Encoding ----------------

    cat_cols = df.select_dtypes(include="object").columns

    if st.button("Apply Encoding"):

        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        st.success("Encoding Applied")

# ---------------- Scaling ----------------

    num_cols = df.select_dtypes(include=np.number).columns

    scale_cols = st.multiselect("Columns for Scaling",num_cols)

    scale_method = st.selectbox(
        "Scaling Method",
        ["Standardization","Normalization"]
    )

    if st.button("Apply Scaling"):

        scaler = StandardScaler() if scale_method=="Standardization" else MinMaxScaler()

        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        st.success("Scaling Applied")

# ---------------- Model Setup ----------------

    st.subheader("Model Setup")

    target = st.selectbox("Target Column",df.columns)

    auto_detect = st.checkbox("Auto Detect Problem Type")

    if auto_detect:

        if df[target].nunique() < 20:
            task = "Classification"
        else:
            task = "Regression"

        st.info(f"Detected Task: {task}")

    else:

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

# ---------------- Split ----------------

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

# ---------------- Hyperparameter Option ----------------

    tune = st.checkbox("Enable Hyperparameter Tuning")

# ---------------- Model Leaderboard ----------------

    st.subheader("Model Leaderboard")

    results=[]
    best_model=None
    best_model_name=None
    best_score=-999

# ---------------- Classification ----------------

    if task=="Classification":

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

        for name,model in models.items():

            if tune and name=="Random Forest":

                params={
                    "n_estimators":[50,100,200],
                    "max_depth":[None,5,10]
                }

                search=RandomizedSearchCV(model,params,n_iter=5,cv=3)

                search.fit(X_train,y_train)

                model=search.best_estimator_

            else:
                model.fit(X_train,y_train)

            preds=model.predict(X_test)

            acc=accuracy_score(y_test,preds)

            cv=cross_val_score(model,X,y,cv=5).mean()

            results.append([name,acc,cv])

            if acc>best_score:
                best_score=acc
                best_model=model
                best_model_name=name

        res=pd.DataFrame(results,columns=["Model","Accuracy","CV Score"])

        st.dataframe(res)

        st.plotly_chart(px.bar(res,x="Model",y="Accuracy"))

# ---------------- Regression ----------------

    else:

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

        st.plotly_chart(px.bar(res,x="Model",y="RMSE"))

# ---------------- Best Model ----------------

    st.subheader("🏆 Best Model")

    st.success(best_model_name)

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

        if hasattr(best_model,"predict_proba"):

            probs=best_model.predict_proba(X_test)[:,1]

            fpr,tpr,_=roc_curve(y_test,probs)

            roc_auc=auc(fpr,tpr)

            fig=go.Figure()

            fig.add_trace(go.Scatter(x=fpr,y=tpr))

            fig.update_layout(title=f"ROC Curve AUC={roc_auc:.2f}")

            st.plotly_chart(fig)

    else:

        st.write("RMSE:",np.sqrt(mean_squared_error(y_test,preds)))
        st.write("MAE:",mean_absolute_error(y_test,preds))
        st.write("R2:",r2_score(y_test,preds))

# ---------------- Learning Curve ----------------

    st.subheader("Learning Curve")

    train_sizes,train_scores,test_scores = learning_curve(
        best_model,X,y,cv=5
    )

    fig=px.line(
        x=train_sizes,
        y=np.mean(test_scores,axis=1),
        labels={"x":"Training Size","y":"Score"}
    )

    st.plotly_chart(fig)

# ---------------- Feature Importance ----------------

    if hasattr(best_model,"feature_importances_"):

        imp=pd.DataFrame({
            "Feature":X.columns,
            "Importance":best_model.feature_importances_
        }).sort_values("Importance",ascending=False)

        st.plotly_chart(px.bar(imp,x="Feature",y="Importance"))

# ---------------- Prediction ----------------

    st.subheader("Prediction")

    user_input={}

    for col in X.columns:
        user_input[col]=st.number_input(col,value=float(X[col].mean()))

    input_df=pd.DataFrame([user_input])

    if st.button("Predict"):
        pred=best_model.predict(input_df)
        st.success(pred[0])

# ---------------- Download Model ----------------

    model_bytes=pickle.dumps(best_model)

    st.download_button(
        "Download Model",
        data=model_bytes,
        file_name="model.pkl"
    )

else:

    st.info("Upload dataset to start AutoML")
