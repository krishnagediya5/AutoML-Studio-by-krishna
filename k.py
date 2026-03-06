import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import accuracy_score, mean_squared_error


st.set_page_config(page_title="AutoML Pro Studio", layout="wide")

st.title("🚀 AutoML Pro Studio")
st.write("Professional End-to-End Machine Learning Platform")

# ---------------------------------------------------
# Upload Dataset
# ---------------------------------------------------

st.sidebar.header("📂 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    st.success("Dataset Loaded Successfully")

    # ---------------------------------------------------
    # EDA SECTION
    # ---------------------------------------------------

    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Dataset Shape:", df.shape)

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Distribution Plot",
            numeric_cols
        )

        fig = px.histogram(df, x=selected_col)

        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) > 1:

        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig = px.imshow(corr, text_auto=True)

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------

    st.sidebar.markdown("---")

    preprocess = st.sidebar.selectbox(
        "⚙️ Preprocessing",
        [
            "None",
            "Fill Missing Values",
            "Remove Duplicates",
            "Encoding",
            "Scaling"
        ]
    )

    if preprocess == "Fill Missing Values":

        cols = st.multiselect("Select Columns", df.columns)

        method = st.selectbox(
            "Method",
            ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill"]
        )

        if st.button("Apply Missing Fill"):

            for col in cols:

                if method == "Mean" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())

                elif method == "Median" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())

                elif method == "Mode":
                    df[col] = df[col].fillna(df[col].mode()[0])

                elif method == "Forward Fill":
                    df[col] = df[col].ffill()

                elif method == "Backward Fill":
                    df[col] = df[col].bfill()

            st.session_state.df = df

            st.success("Missing Values Handled")

    elif preprocess == "Remove Duplicates":

        st.write("Duplicate Rows:", df.duplicated().sum())

        if st.button("Remove Duplicates"):

            df = df.drop_duplicates()

            st.session_state.df = df

            st.success("Duplicates Removed")

    elif preprocess == "Encoding":

        cat_cols = df.select_dtypes(include="object").columns

        selected = st.multiselect("Categorical Columns", cat_cols)

        if st.button("Apply Encoding"):

            for col in selected:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

            st.session_state.df = df

            st.success("Encoding Applied")

    elif preprocess == "Scaling":

        num_cols = df.select_dtypes(include=np.number).columns

        selected = st.multiselect("Numeric Columns", num_cols)

        method = st.selectbox(
            "Scaling Method",
            ["Standardization", "Normalization"]
        )

        if st.button("Apply Scaling"):

            if method == "Standardization":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            df[selected] = scaler.fit_transform(df[selected])

            st.session_state.df = df

            st.success("Scaling Applied")

    # ---------------------------------------------------
    # FEATURE SELECTION
    # ---------------------------------------------------

    st.sidebar.markdown("---")

    feature_select = st.sidebar.selectbox(
        "🎯 Feature Selection",
        ["None", "SelectKBest"]
    )

    if feature_select == "SelectKBest":

        target = st.selectbox("Target Column", df.columns)

        task = st.radio(
            "Task Type",
            ["Classification", "Regression"]
        )

        k = st.slider(
            "Top K Features",
            1,
            len(df.columns) - 1,
            3
        )

        if st.button("Run Feature Selection"):

            X = df.drop(columns=[target]).select_dtypes(include=np.number)
            y = df[target]

            selector = SelectKBest(
                f_classif if task == "Classification" else f_regression,
                k=k
            )

            X_new = selector.fit_transform(X, y)

            cols = X.columns[selector.get_support()]

            df_selected = pd.concat(
                [pd.DataFrame(X_new, columns=cols), y.reset_index(drop=True)],
                axis=1
            )

            st.session_state.df_selected = df_selected

            st.success("Feature Selection Completed")

            st.dataframe(df_selected)

    # ---------------------------------------------------
    # MODEL TRAINING
    # ---------------------------------------------------

    st.sidebar.markdown("---")

    model_menu = st.sidebar.selectbox(
        "🤖 Model Training",
        ["None", "AutoML"]
    )

    if model_menu == "AutoML":

        if "df_selected" not in st.session_state:

            st.warning("Run Feature Selection First")

        else:

            df_sel = st.session_state.df_selected

            target = st.selectbox("Target", df_sel.columns)

            task = st.radio(
                "Task",
                ["Classification", "Regression"]
            )

            X = df_sel.drop(columns=[target])
            y = df_sel[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            st.subheader("🏆 Model Leaderboard")

            results = []

            if task == "Classification":

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "KNN": KNeighborsClassifier()
                }

                for name, model in models.items():

                    model.fit(X_train, y_train)

                    pred = model.predict(X_test)

                    acc = accuracy_score(y_test, pred)

                    results.append([name, acc])

                res = pd.DataFrame(results, columns=["Model", "Accuracy"])

                st.dataframe(res)

                best_model_name = res.sort_values(
                    "Accuracy",
                    ascending=False
                ).iloc[0]["Model"]

            else:

                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "KNN": KNeighborsRegressor()
                }

                for name, model in models.items():

                    model.fit(X_train, y_train)

                    pred = model.predict(X_test)

                    rmse = np.sqrt(mean_squared_error(y_test, pred))

                    results.append([name, rmse])

                res = pd.DataFrame(results, columns=["Model", "RMSE"])

                st.dataframe(res)

                best_model_name = res.sort_values("RMSE").iloc[0]["Model"]

            best_model = models[best_model_name]

            best_model.fit(X_train, y_train)

            st.success(f"🏆 Best Model: {best_model_name}")

            # ---------------------------------------------------
            # DOWNLOAD MODEL
            # ---------------------------------------------------

            model_bytes = pickle.dumps(best_model)

            st.download_button(
                label="💾 Download Model",
                data=model_bytes,
                file_name="trained_model.pkl"
            )

            # ---------------------------------------------------
            # PREDICTION
            # ---------------------------------------------------

            st.subheader("🔮 Make Prediction")

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

    st.info("Upload a CSV dataset to start AutoML")
