import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from langflow.custom import Component
from langflow.io import DropdownInput, HandleInput, MessageTextInput, Output
from langflow.schema import DataFrame


class DataScalerComponent(Component):
    display_name = "Data Scaler"
    description = "Scale features using different scaling methods"
    documentation = "https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing"
    icon = "ScikitLearn"
    scaled_data = None
    scaler_instance = None

    SCALER_MAPPING = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
        "MaxAbsScaler": MaxAbsScaler,
    }

    inputs = [
        HandleInput(
            name="dataset",
            display_name="Dataset",
            info="The dataset to scale",
            input_types=["DataFrame"],
        ),
        DropdownInput(
            name="scaler_type",
            display_name="Scaler Type",
            options=list(SCALER_MAPPING.keys()),
            value="StandardScaler",
            info="Type of scaling to apply",
        ),
        MessageTextInput(
            name="target_column",
            display_name="Target Column",
            info="The column name of the target variable",
            value="target",
        ),
    ]

    outputs = [
        Output(display_name="Scaled DataFrame", name="scaled_dataframe", method="get_scaled_dataframe"),
        Output(display_name="Scaler Object", name="scaler_object", method="get_scaler_object"),
    ]

    def scale_data(self):
        if not hasattr(self, "dataset"):
            msg = "No dataset provided. Please connect a dataset component."
            raise ValueError(msg)

        if not isinstance(self.dataset, DataFrame):
            msg = "The dataset is not a DataFrame. Please connect a DataFrame component."
            raise TypeError(msg)

        # Create the appropriate scaler instance
        scaler_class = self.SCALER_MAPPING.get(self.scaler_type)
        if scaler_class is None:
            msg = f"Invalid scaler type: {self.scaler_type}"
            raise ValueError(msg)

        # Initialize and fit the scaler
        if self.target_column in self.dataset.columns:
            self.dataset_features = self.dataset.drop(self.target_column, axis=1)
        else:
            msg = "Target column not found in dataset"
            raise ValueError(msg)
        self.scaler_instance = scaler_class()
        scaled_data = self.scaler_instance.fit_transform(self.dataset_features)
        scaled_data = self.scaler_instance.fit_transform(self.dataset_features)
        # Add the target column back to the scaled data
        scaled_data = pd.DataFrame(scaled_data, columns=self.dataset_features.columns)
        concat_data = pd.concat([scaled_data, self.dataset[self.target_column]], axis=1)

        # Convert to DataFrame with original column names
        scale_df = pd.DataFrame(concat_data, columns=self.dataset.columns)
        self.scaled_data = DataFrame(scale_df)

    def get_scaled_dataframe(self) -> DataFrame:
        if self.scaled_data is None:
            self.scale_data()
        self.status = self.scaled_data
        return self.scaled_data

    def get_scaler_object(self) -> BaseEstimator:
        if self.scaler_instance is None:
            self.scale_data()
        return self.scaler_instance
