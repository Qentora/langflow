from sklearn.model_selection import train_test_split

from langflow.custom import Component
from langflow.field_typing.range_spec import RangeSpec
from langflow.io import HandleInput, IntInput, MessageTextInput, Output, SliderInput
from langflow.schema import DataFrame


class TrainTestSplitComponent(Component):
    display_name = "Train Test Split"
    description = "Split arrays or matrices into random train and test subsets"
    documentation = "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"
    icon = "ScikitLearn"
    TrainSplit = None
    TestSplit = None

    inputs = [
        HandleInput(
            name="dataset",
            display_name="Dataset",
            info="The dataset to split",
            input_types=["DataFrame"],
        ),
        SliderInput(
            name="test_size",
            display_name="Test Size",
            value=0.25,
            range_spec=RangeSpec(min=0, max=0.9, step=0.01),
        ),
        IntInput(
            name="random_state",
            display_name="Random State",
            value=42,
            info="Controls the shuffling applied to the data before applying the split",
        ),
        MessageTextInput(
            name="target_column",
            display_name="Target Column",
            info="The column name of the target variable",
            value="target",
        ),
    ]

    outputs = [
        Output(display_name="Train DataFrame", name="split_dataframe_train", method="split_dataframe_train"),
        Output(display_name="Test DataFrame", name="split_dataframe_test", method="split_dataframe_test"),
    ]

    def split_dataframe_base(self):
        if not hasattr(self, "dataset"):
            msg = "No dataset provided. Please connect a dataset component."
            raise ValueError(msg)
        # split the dataset into train and test
        if not isinstance(self.dataset, DataFrame):
            msg = "The dataset is not a DataFrame. Please connect a DataFrame component."
            raise TypeError(msg)

        if self.target_column not in self.dataset.columns:
            msg = f"Error: The target column '{self.target_column}' does not exist in the dataset."
            raise ValueError(msg)

        train_df, test_df = train_test_split(self.dataset, test_size=self.test_size, random_state=self.random_state)
        self.TrainSplit = train_df
        self.TestSplit = test_df

    def split_dataframe_train(self) -> DataFrame:
        self.split_dataframe_base()
        return DataFrame(self.TrainSplit)

    def split_dataframe_test(self) -> DataFrame:
        self.split_dataframe_base()
        return DataFrame(self.TestSplit)
