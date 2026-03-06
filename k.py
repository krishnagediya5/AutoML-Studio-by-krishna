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

st.title("🚀 AutoML Studio")
st.write("End-to-End Machine Learning Platform")

st.sidebar.header("📂 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df
    st.success("Dataset Loaded Successfully")

    # ---------------- PREPROCESSING ---------------- #

    preprocess_menu = st.sidebar.selectbox(
        "⚙️ Preprocessing",
        [
            "Data Preview",
            "Missing Value Count",
            "Fill Missing Values",
            "Duplicate Rows",
            "Encoding",
            "Scaling",
            "Final Dataset"
        ]
    )

    if preprocess_menu == "Data Preview":
        st.dataframe(df)

    elif preprocess_menu == "Missing Value Count":
        st.write(df.isnull().sum())

    elif preprocess_menu == "Fill Missing Values":

        cols = st.multiselect("Select Column(s)", df.columns)

        method = st.selectbox(
            "Method",
            ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill"]
        )

        if st.button("Apply Missing Value Method"):

            if len(cols) == 0:
                st.warning("Please select at least one column")

            else:
                for col in cols:

                    if method == "Mean":
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].mean())

                    elif method == "Median":
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median())

                    elif method == "Mode":
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col] = df[col].fillna(mode_val.iloc[0])

                    elif method == "Forward Fill":
                        df[col] = df[col].ffill()

                    elif method == "Backward Fill":
                        df[col] = df[col].bfill()

                st.session_state.df = df
                st.success("Missing values handled successfully")

    elif preprocess_menu == "Duplicate Rows":

        st.write("Duplicate rows:", df.duplicated().sum())

        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.session_state.df = df
            st.success("Duplicates removed")

    elif preprocess_menu == "Encoding":

        cat_cols = df.select_dtypes(include="object").columns

        selected = st.multiselect("Select columns to encode", cat_cols)

        if st.button("Apply Encoding"):

            for col in selected:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

            st.session_state.df = df
            st.success("Encoding applied")

    elif preprocess_menu == "Scaling":

        num_cols = df.select_dtypes(include=np.number).columns

        selected = st.multiselect("Select columns to scale", num_cols)

        if st.button("Apply Scaling"):

            scaler = StandardScaler()

            df[selected] = scaler.fit_transform(df[selected])

            st.session_state.df = df
            st.success("Scaling applied")

    elif preprocess_menu == "Final Dataset":
        st.dataframe(df)

    # ---------------- FEATURE SELECTION ---------------- #

    st.sidebar.markdown("---")

    feature_menu = st.sidebar.selectbox(
        "🎯 Feature Selection",
        ["None", "SelectKBest"]
    )

    if feature_menu == "SelectKBest":

        target = st.selectbox("Target Column", df.columns)

        task = st.radio(
            "Task Type",
            ["Classification", "Regression"]
        )

        k = st.slider("Top K Features", 1, len(df.columns) - 1, 3)

        if st.button("Run Feature Selection"):

            X = df.drop(columns=[target]).select_dtypes(include=np.number)
            y = df[target]

            selector = SelectKBest(
                f_classif if task == "Classification" else f_regression,
                k=k
            )

            X_new = selector.fit_transform(X, y)

            selected_cols = X.columns[selector.get_support()]

            df_selected = pd.concat(
                [pd.DataFrame(X_new, columns=selected_cols), y.reset_index(drop=True)],
                axis=1
            )

            st.session_state.df_selected = df_selected

            st.success("Feature selection completed")

            st.dataframe(df_selected)

    # ---------------- MODEL TRAINING ---------------- #

    st.sidebar.markdown("---")

    model_menu = st.sidebar.selectbox(
        "🤖 Model",
        ["None", "AutoModel"]
    )

    if model_menu == "AutoModel":

        if "df_selected" not in st.session_state:
            st.warning("Please run feature selection first")

        else:

            df_sel = st.session_state.df_selected

            target = st.selectbox("Target Column", df_sel.columns)

            task = st.radio(
                "Task Type",
                ["Classification", "Regression"]
            )

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

            st.success(f"🏆 Best Model Selected: {best_model_name}")

            # ---------------- PREDICTION ---------------- #

            st.subheader("🔮 Predict Using User Input")

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

    st.info("Upload a CSV dataset to begin.")

    st.markdown("""
### Features

✔ Upload dataset  
✔ Handle missing values (Mean / Median / Mode / ffill / bfill)  
✔ Remove duplicates  
✔ Encode categorical data  
✔ Scale numeric features  
✔ Feature selection  
✔ Auto model comparison  
✔ Prediction
""")
