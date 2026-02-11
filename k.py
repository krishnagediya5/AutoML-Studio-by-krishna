import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, mean_squared_error


st.set_page_config(page_title="AutoML Studio", layout="wide")


st.markdown("""
<div style='text-align: center; padding-top: 40px; padding-bottom: 20px;'>
<h1 style='font-size: 50px; font-weight: 700;'>🚀 AutoML Studio</h1>
<h3 style='color: #9CA3AF; font-weight: 400;'>An End-to-End Machine Learning Platform</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.info("📂 Upload your CSV dataset from the sidebar to begin.")


st.sidebar.header("📂 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="file_upload")

if file is not None:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df
    st.success("✅ Dataset Loaded Successfully")

   
    preprocess_menu = st.sidebar.selectbox(
        "⚙️ Preprocessing",
        ["Data Preview", "Missing Value Count", "Fill Missing Values",
         "Duplicate Rows", "Encoding", "Scaling", "Final Dataset"],
        key="preprocess_menu"
    )

    if preprocess_menu == "Fill Missing Values":
        col = st.selectbox("Column", df.columns, key="fill_col")
        method = st.selectbox(
            "Method",
            ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill"],
            key="fill_method"
        )

        if st.button("Apply", key="fill_apply"):
            if method == "Mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "Median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == "Forward Fill":
                df[col].fillna(method="ffill", inplace=True)
            elif method == "Backward Fill":
                df[col].fillna(method="bfill", inplace=True)

            st.session_state.df = df
            st.success("✅ Missing values handled")

    elif preprocess_menu == "Data Preview":
        st.dataframe(df)

    elif preprocess_menu == "Missing Value Count":
        st.write(df.isnull().sum())

    elif preprocess_menu == "Duplicate Rows":
        st.write("Duplicate rows:", df.duplicated().sum())
        if st.button("Remove Duplicates", key="remove_duplicates"):
            df.drop_duplicates(inplace=True)
            st.session_state.df = df
            st.success("✅ Duplicates removed")

    elif preprocess_menu == "Encoding":
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        st.session_state.df = df
        st.success("✅ Encoding applied")

    elif preprocess_menu == "Scaling":
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = StandardScaler().fit_transform(df[num_cols])
        st.session_state.df = df
        st.success("✅ Scaling applied")

    elif preprocess_menu == "Final Dataset":
        st.dataframe(df)

    
    st.sidebar.markdown("---")
    feature_menu = st.sidebar.selectbox(
        "🎯 Feature Selection",
        ["None", "SelectKBest"],
        key="feature_menu"
    )

    if feature_menu == "SelectKBest":
        target = st.selectbox("Target Column", df.columns, key="fs_target")
        task = st.radio("Task Type", ["Classification", "Regression"], key="fs_task")
        k = st.slider("Top K Features", 1, len(df.columns) - 1,
                      min(5, len(df.columns) - 1), key="fs_k")

        if st.button("Run Feature Selection", key="fs_run"):
            X = df.drop(columns=[target]).select_dtypes(include="number")
            y = df[target]

            selector = SelectKBest(
                f_classif if task == "Classification" else f_regression,
                k=k
            )

            X_new = selector.fit_transform(X, y)
            cols = X.columns[selector.get_support()]

            st.session_state.df_selected = pd.concat(
                [pd.DataFrame(X_new, columns=cols), y.reset_index(drop=True)],
                axis=1
            )

            st.success("✅ Feature selection completed")
            st.dataframe(st.session_state.df_selected)

    
    st.sidebar.markdown("---")
    model_menu = st.sidebar.selectbox(
        "🤖 Model",
        ["None", "AutoModel"],
        key="model_menu"
    )

    if model_menu == "AutoModel":

        if "df_selected" not in st.session_state:
            st.warning("⚠️ Please run feature selection first")
        else:
            st.subheader("📊 AutoModel – Compare All Models")

            df_sel = st.session_state.df_selected
            target = st.selectbox("Target Column", df_sel.columns, key="auto_target")
            task = st.radio("Task Type", ["Classification", "Regression"], key="auto_task")

            X = df_sel.drop(columns=[target])
            y = df_sel[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            results = []

            if task == "Classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "KNN": KNeighborsClassifier(),
                    "SVM": SVC(probability=True),
                    "Naive Bayes": GaussianNB(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "AdaBoost": AdaBoostClassifier(),
                    "Extra Trees": ExtraTreesClassifier()
                }

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    acc = accuracy_score(y_test, model.predict(X_test))
                    results.append([name, acc])

                res = pd.DataFrame(results, columns=["Model", "Accuracy"])
                st.dataframe(res)
                best_model_name = res.sort_values("Accuracy", ascending=False).iloc[0]["Model"]

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
                    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
                    results.append([name, rmse])

                res = pd.DataFrame(results, columns=["Model", "RMSE"])
                st.dataframe(res)
                best_model_name = res.sort_values("RMSE").iloc[0]["Model"]

            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            st.success(f"🏆 Best Model Selected: {best_model_name}")

            st.markdown("---")
            st.subheader("🔮 Predict Using User Input")

            user_input = {}
            for col in X.columns:
                user_input[col] = st.number_input(
                    f"Enter {col}",
                    value=float(X[col].mean()),
                    key=f"user_{col}"
                )

            input_df = pd.DataFrame([user_input])

            if st.button("Predict", key="predict_button"):
                prediction = best_model.predict(input_df)

                if task == "Classification":
                    st.success(f"🎯 Predicted Class: {prediction[0]}")
                else:
                    st.success(f"📈 Predicted Value: {prediction[0]:.4f}")

else:
    st.markdown("## 📌 User Description")
    st.markdown("""
- 📊 Upload and preview dataset  
- 🧹 Clean missing values and duplicates  
- 🔤 Encode categorical variables  
- ⚖️ Scale numerical features  
- 🎯 Select best features  
- 🤖 Compare ML models automatically  
- 🔮 Make predictions easily  

AutoML Studio simplifies the complete ML workflow in one dashboard.
""")
