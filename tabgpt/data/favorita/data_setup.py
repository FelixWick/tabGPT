from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


class FavoritaData(DataFrameLoader):
    def __init__(self, task_description='favorita store sales'):
        super().__init__(task_description)

    def setup(self, last_nrows=50_000):
        col_dict = {
        "store_nbr": "unique store id",
        "item_nbr": "unique item number",
        "unit_sales": "Sales",
        "onpromotion": "product is in promotion",
        "family": "product type",
        "class": "product class",
        "perishable": "product is perishable",
        "transferred": "holiday transferred to other day",
        "day_of_week": "day of the week",
        }

        df_train = pd.read_csv(
            os.path.join(self.current_dir,"train.csv"),
            parse_dates=["date"],
            skiprows=range(1, 124600000),
        )  # result ~ 800.000 rows
        items = pd.read_csv(os.path.join(self.current_dir,"items.csv"))
        df_holiday_events = pd.read_csv(os.path.join(self.current_dir,"holidays_events.csv"))
        df_holiday_events["date"] = pd.to_datetime(df_holiday_events["date"])
        df = df_train.merge(items, how="left", on="item_nbr")
        df = df.merge(df_holiday_events, how="left", on="date")
        df["month"] = df["date"].dt.month_name(locale="English")
        df["year"] = df["date"].dt.year
        df["day_of_week"] = df["date"].dt.day_name()
        df.drop(["date"], axis=1, inplace=True)

        df = df[df["unit_sales"] >= 0]

        df.drop(["id"], axis=1, inplace=True)

        df = df.assign(
            onpromotion=lambda df: df.onpromotion.map({False: "no", True: "yes"})
        ).assign(perishable=lambda df: df.perishable.map({0: "no", 1: "yes"}))

        df = df.rename(columns=col_dict)

        df = df.iloc[-last_nrows:]  # last
        train_rows = int(0.95 * last_nrows)
        df_train = df[:train_rows]
        df_val = df[train_rows:]

        df_train["target"] = np.log1p(df_train["unit_sales"])
        df_val['target'] = df_val['unit_sales']

        df_train.drop(["unit_sales"], axis=1, inplace=True)
        df_val.drop(["unit_sales"], axis=1, inplace=True)


        categorical_features = df_train.columns[df_train.columns != "target"].tolist()
        numerical_features = []

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_features = len(categorical_features + numerical_features)
        self.target_column = 'target'





