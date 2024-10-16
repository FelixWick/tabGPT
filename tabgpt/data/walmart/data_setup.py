from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from tabgpt.utils import stratify


class WalmartData(DataFrameLoader):
    def __init__(self, task_description='Walmart store products'):
        super().__init__(task_description)

    def setup(self, last_nrows=50_000):
        col_dict = {
        "Dept": "department number",
        "Store": "store number",
        "Fuel_Price": "cost of fuel",
        "CPI": "consumer price index",
        "Unemployment": "unemployment rate",
        "IsHoliday": "special holiday week",
        "Size": "store size (in square feet)",
        "Temperature": "average temperature (in Fahrenheit)",
        }

        dataset = pd.read_csv(os.path.join(self.current_dir,"train.csv"))
        features = pd.read_csv(os.path.join(self.current_dir,"features.csv"))
        stores = pd.read_csv(os.path.join(self.current_dir,"stores.csv"))
        features = features.merge(stores, how="inner", on="Store")
        df = (
            dataset.merge(features, how="inner", on=["Store", "Date", "IsHoliday"])
            .sort_values(by=["Store", "Dept", "Date"])
            .reset_index(drop=True)
        )

        df["Date"] = pd.to_datetime(df["Date"])

        df["month"] = df["Date"].dt.month_name()
        df["year"] = df["Date"].dt.year
        df["week"] = df["Date"].dt.isocalendar().week

        df.drop(
            [
                "Date",
                "MarkDown1",
                "MarkDown2",
                "MarkDown3",
                "MarkDown4",
                "MarkDown5",
                "Fuel_Price",
            ],
            axis=1,
            inplace=True,
        )

        df = df.assign(IsHoliday=lambda df: df.IsHoliday.map({False: "no", True: "yes"}))
        df = df[df["Weekly_Sales"] > 0]

        df = stratify(df,groupby_cols=['Store', 'Dept'])
        df = df.rename(columns=col_dict)

        df_train, df_val = train_test_split(df,test_size=0.1,stratify=df[["store number","department number"]])
        

        numerical_features = [
            "average temperature (in Fahrenheit)",
            "consumer price index",
            "unemployment rate",
            "store size (in square feet)",
        ]
        categorical_features = df_train.columns[
            (df_train.columns != "Weekly_Sales") & ~(df_train.columns.isin(numerical_features))
        ].tolist()

        self.setup_scaler(numerical_features)
        self.scale_columns(df_train, mode='train')
        self.scale_columns(df_val)

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.set_target_column(main_target='Weekly_Sales')





