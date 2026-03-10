from sklearn import datasets

from langflow.custom import Component
from langflow.io import DropdownInput, Output
from langflow.schema import Data, DataFrame


class SklearnDatasetComponent(Component):
    display_name = "Sklearn Dataset"
    description = "Load a dataset from scikit-learn"
    documentation = "https://scikit-learn.org/stable/datasets.html"
    icon = "ScikitLearn"

    AVAILABLE_DATASETS = {
        "iris": datasets.load_iris,
        "digits": datasets.load_digits,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "diabetes": datasets.load_diabetes,
    }

    inputs = [
        DropdownInput(
            name="dataset_name",
            display_name="Dataset",
            options=list(AVAILABLE_DATASETS.keys()),
            value="iris",
            info="Select a dataset from scikit-learn",
        ),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="load_dataset_as_data"),
        Output(display_name="DataFrame", name="dataframe", method="load_dataset_as_dataframe"),
    ]

    def load_dataset_as_dataframe(self) -> DataFrame:
        dataset_name = self.dataset_name
        if dataset_name not in self.AVAILABLE_DATASETS:
            msg = f"Dataset {dataset_name} not found"
            raise ValueError(msg)

        dataset = self.AVAILABLE_DATASETS[dataset_name](as_frame=True)
        # Create a dictionary with both data and target
        data_dict = dataset.data.copy()
        data_dict["target"] = dataset.target
        return DataFrame(data=data_dict)

    def load_dataset_as_data(self) -> Data:
        dataset_name = self.dataset_name
        if dataset_name not in self.AVAILABLE_DATASETS:
            msg = f"Dataset {dataset_name} not found"
            raise ValueError(msg)

        dataset = self.AVAILABLE_DATASETS[dataset_name](as_frame=True)
        data_dict = {
            "data": dataset.data.to_dict(),
            "target": dataset.target.to_dict() if hasattr(dataset.target, "to_dict") else dataset.target.tolist(),
            "feature_names": dataset.feature_names if hasattr(dataset, "feature_names") else None,
            "target_names": dataset.target_names if hasattr(dataset, "target_names") else None,
            "DESCR": dataset.DESCR if hasattr(dataset, "DESCR") else None,
            "dataset_name": dataset_name,
        }

        return Data(**data_dict)
