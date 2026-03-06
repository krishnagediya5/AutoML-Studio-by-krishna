import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

try:
    import shap
    SHAP_AVAILABLE=True
except:
    SHAP_AVAILABLE=False

from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_classif,f_regression
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV

from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error,r2_score

from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor

st.set_page_config(page_title="AutoML Pro Studio",layout="wide")

st.title("🚀 AutoML Pro Studio v2")

# ================= Upload =================

st.sidebar.header("Upload Dataset")

file=st.sidebar.file_uploader("Upload CSV",type=["csv"])

if file:

    df=pd.read_csv(file)

    st.success("Dataset Loaded")

# ================= Preview =================

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# ================= Correlation =================

    st.subheader("Correlation Heatmap")

    corr=df.corr(numeric_only=True)

    fig=px.imshow(corr,text_auto=True)

    st.plotly_chart(fig)

# ================= EDA =================

    numeric_cols=df.select_dtypes(include=np.number).columns

    if len(numeric_cols)>0:

        col=st.selectbox("Distribution Column",numeric_cols)

        fig=px.histogram(df,x=col)

        st.plotly_chart(fig)

# ================= Preprocessing =================

    st.subheader("Missing Value Handling")

    fill_cols=st.multiselect("Columns",df.columns)

    method=st.selectbox("Method",["Mean","Median","Mode"])

    if st.button("Apply Fill"):

        for col in fill_cols:

            if method=="Mean":
                df[col]=df[col].fillna(df[col].mean())

            elif method=="Median":
                df[col]=df[col].fillna(df[col].median())

            else:
                df[col]=df[col].fillna(df[col].mode()[0])

# ================= Encoding =================

    cat_cols=df.select_dtypes(include="object").columns

    encode_cols=st.multiselect("Encode Columns",cat_cols)

    if st.button("Apply Encoding"):

        for col in encode_cols:
            df[col]=LabelEncoder().fit_transform(df[col].astype(str))

# ================= Scaling =================

    num_cols=df.select_dtypes(include=np.number).columns

    scale_cols=st.multiselect("Scale Columns",num_cols)

    scale_method=st.selectbox("Scaling",["Standard","MinMax"])

    if st.button("Apply Scaling"):

        scaler=StandardScaler() if scale_method=="Standard" else MinMaxScaler()

        df[scale_cols]=scaler.fit_transform(df[scale_cols])

# ================= Target =================

    target=st.selectbox("Target Column",df.columns)

    X=df.drop(columns=[target])
    y=df[target]

# ================= Auto Task =================

    if y.dtype=="object" or len(y.unique())<=10:
        task="Classification"
    else:
        task="Regression"

    st.write("Detected Task:",task)

# ================= Feature Engineering =================

    if st.checkbox("Auto Feature Engineering"):

        new_features={}

        for col in X.select_dtypes(include=np.number).columns:

            new_features[f"{col}_square"]=X[col]**2
            new_features[f"{col}_log"]=np.log1p(np.abs(X[col]))

        X=pd.concat([X,pd.DataFrame(new_features)],axis=1)

        st.success("Features Generated")

# ================= Feature Selection =================

    k=st.slider("Top K Features",1,X.shape[1],min(5,X.shape[1]))

    selector=SelectKBest(
        f_classif if task=="Classification" else f_regression,
        k=k
    )

    X_new=selector.fit_transform(X,y)

    selected_features=X.columns[selector.get_support()]

    X=pd.DataFrame(X_new,columns=selected_features)

# ================= Train Test =================

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )

# ================= Models =================

    st.subheader("Model Leaderboard")

    results=[]
    best_model=None
    best_model_name=None

# ================= Classification =================

    if task=="Classification":

        best_score=0

        models={

        "LogisticRegression":LogisticRegression(max_iter=1000),
        "RandomForest":RandomForestClassifier(),
        "ExtraTrees":ExtraTreesClassifier(),
        "GradientBoost":GradientBoostingClassifier(),
        "DecisionTree":DecisionTreeClassifier(),
        "KNN":KNeighborsClassifier(),
        "SVM":SVC(probability=True),
        "NaiveBayes":GaussianNB(),

        "XGBoost":XGBClassifier(eval_metric="logloss"),
        "LightGBM":LGBMClassifier(),
        "CatBoost":CatBoostClassifier(verbose=0)

        }

        for name,model in models.items():

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

        fig=px.bar(res,x="Model",y="Accuracy")

        st.plotly_chart(fig)

# ================= Regression =================

    else:

        best_score=float("inf")

        models={

        "LinearRegression":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(),
        "RandomForest":RandomForestRegressor(),
        "ExtraTrees":ExtraTreesRegressor(),
        "GradientBoost":GradientBoostingRegressor(),
        "DecisionTree":DecisionTreeRegressor(),
        "KNN":KNeighborsRegressor(),
        "SVR":SVR(),

        "XGBoost":XGBRegressor(),
        "LightGBM":LGBMRegressor(),
        "CatBoost":CatBoostRegressor(verbose=0)

        }

        for name,model in models.items():

            model.fit(X_train,y_train)

            preds=model.predict(X_test)

            rmse=np.sqrt(mean_squared_error(y_test,preds))

            cv=cross_val_score(
                model,X,y,cv=5,
                scoring="neg_mean_squared_error"
            ).mean()

            results.append([name,rmse,cv])

            if rmse<best_score:
                best_score=rmse
                best_model=model
                best_model_name=name

        res=pd.DataFrame(results,columns=["Model","RMSE","CV Score"])

        st.dataframe(res)

        fig=px.bar(res,x="Model",y="RMSE")

        st.plotly_chart(fig)

# ================= Best Model =================

    st.subheader("Best Model")

    st.success(best_model_name)

# ================= Feature Importance =================

    if hasattr(best_model,"feature_importances_"):

        imp=pd.DataFrame({
        "Feature":X.columns,
        "Importance":best_model.feature_importances_
        })

        fig=px.bar(imp.sort_values("Importance",ascending=False),
        x="Feature",y="Importance")

        st.plotly_chart(fig)

# ================= SHAP =================

    if SHAP_AVAILABLE:

        st.subheader("Explainable AI (SHAP)")

        try:

            explainer=shap.Explainer(best_model,X_train)

            shap_values=explainer(X_test)

            fig=shap.plots.bar(shap_values,show=False)

            st.pyplot(fig)

        except:

            st.info("SHAP not supported")

# ================= Hyperparameter Tuning =================

    if st.button("Run Hyperparameter Tuning"):

        param_grid={
        "n_estimators":[100,200,300],
        "max_depth":[None,5,10]
        }

        tuner=RandomizedSearchCV(
        RandomForestRegressor() if task=="Regression" else RandomForestClassifier(),
        param_grid,
        n_iter=5,
        cv=3,
        n_jobs=-1
        )

        tuner.fit(X_train,y_train)

        best_model=tuner.best_estimator_

        st.success("Tuning Completed")

# ================= Download =================

    model_bytes=pickle.dumps(best_model)

    st.download_button(
    "Download Model",
    data=model_bytes,
    file_name="best_model.pkl"
    )

# ================= Prediction =================

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
