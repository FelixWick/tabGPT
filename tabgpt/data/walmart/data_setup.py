from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


class WalmartData(DataFrameLoader):
    def __init__(self, task_description='Walmart weekly store sales'):
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

        df["month"] = df["Date"].dt.month_name(locale="English")
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
        
        df = df.rename(columns=col_dict)

        df = df.iloc[-last_nrows:]

        train_rows = int(0.95 * last_nrows)
        df_train = df[:train_rows]
        df_val = df[train_rows:]

        df_train["target"] = np.log1p(df_train["Weekly_Sales"])
        df_val["target"] = df_val["Weekly_Sales"]

        df_train.drop(["Weekly_Sales"], axis=1, inplace=True)
        df_val.drop(["Weekly_Sales"], axis=1, inplace=True)

        numerical_features = [
            "average temperature (in Fahrenheit)",
            "unemployment rate",
            "store size (in square feet)",
        ]
        categorical_features = df_train.columns[
            (df_train.columns != "target") & ~(df_train.columns.isin(numerical_features))
        ].tolist()

        num_max = df_train[numerical_features].abs().max()
        df_train[numerical_features] = df_train[numerical_features] / num_max
        df_val[numerical_features] = df_val[numerical_features] / num_max

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_features = len(categorical_features + numerical_features)
        self.target_column = 'target'





