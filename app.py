import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="California Housing Price Prediction",
    page_icon="✨",
    initial_sidebar_state="expanded",
)

with st.container():
    st.header("California Housing Price Prediction")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join("model.pkl")
PIPELINE_FILE = os.path.join("pipeline.pkl")
HOUSING = os.path.join("housing.csv")

with st.container():
    with st.form("house_prediction_form"):
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            longitude = st.number_input(label="longitude", value=-121.46, min_value=-125.00, max_value=-110.00)
            latitude = st.number_input(label="latitude", value=38.52, min_value=30.00, max_value=45.00)
        with col2:
            hma = st.number_input(label="Housing Median Age", value=29.0, min_value=1.00, max_value=55.00)
            tr = st.number_input(label="Total Rooms", value=3873.0, min_value=1.00, max_value=40000.00)
        with col3:
            tb = st.number_input(label="Total Bedrooms", value=797.0, min_value=3.00, max_value=6500.00)
            population = st.number_input(label="Population", value=2237.0, min_value=2.00, max_value=36000.00)
        with col4:
            households = st.number_input(label="Households", value=706.0, min_value=1.00, max_value=6500.00)
            mi = st.number_input(label="Median Income", value=2.1736, min_value=0.3, max_value=16.00)
        op = st.selectbox("Ocean Proximity",("NEAR BAY", "INLAND", "<1H OCEAN", "NEAR OCEAN"), index=1)
        submit = st.form_submit_button("Predict House Value")
        if submit:
            input_data = pd.DataFrame([{
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": hma,
                "total_rooms": tr,
                "total_bedrooms": tb,
                "population": population,
                "households": households,
                "median_income": mi,
                "ocean_proximity": op 
            }])

            def build_pipeline(num_attribs, cat_attribs):
                # 5. Pipelines
                # Numerical pipeline
                num_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                    ])
                # Categorical pipeline
                cat_pipeline = Pipeline([
                    # ("ordinal", OrdinalEncoder()) # Use this if you prefer ordinal encoding
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ])
                # Full pipeline
                full_pipeline = ColumnTransformer([
                    ("num", num_pipeline, num_attribs),
                    ("cat", cat_pipeline, cat_attribs),
                    ])
                return full_pipeline

            if not os.path.exists(MODEL_FILE):
                # 1. Load the data
                housing = pd.read_csv(HOUSING)
                # 2. Create a stratified test set based on income category
                housing["income_cat"] = pd.cut(
                    housing["median_income"],
                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                    labels=[1, 2, 3, 4, 5]
                    )
                split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                for train_index, test_index in split.split(housing, housing["income_cat"]):
                    housing.loc[test_index].drop("income_cat", axis=1)
                    housing = housing.loc[train_index].drop("income_cat", axis=1)

                housing_labels = housing["median_house_value"].copy()
                housing_features = housing.drop("median_house_value", axis=1)

                # 4. Separate numerical and categorical columns
                num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
                cat_attribs = ["ocean_proximity"]

                pipeline = build_pipeline(num_attribs, cat_attribs)
                # 6. Transform the data
                housing_prepared = pipeline.fit_transform(housing_features)

                # housing_prepared is now a NumPy array ready for training
                model = RandomForestRegressor(random_state=42)
                model.fit(housing_prepared, housing_labels)

                joblib.dump(model, MODEL_FILE)
                joblib.dump(pipeline, PIPELINE_FILE)

            else:
                # Load Model File & Pipeline file.
                model = joblib.load(MODEL_FILE)
                pipeline = joblib.load(PIPELINE_FILE)

                transformed_input = pipeline.transform(input_data)
                predictions = model.predict(transformed_input)[0]
                input_data['median_house_value'] = predictions
                st.success(f"Predicted House Value: ${predictions:,.2f}")