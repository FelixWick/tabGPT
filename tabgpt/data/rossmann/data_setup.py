from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


class RossmannData(DataFrameLoader):
    def __init__(self, task_description='Rossmann store sales'):
        super().__init__(task_description)

    def setup(self, last_nrows=50_000):
        col_dict = {
        "Store": "unique store Id",
        "Customers": "number of customers",
        "DayOfWeek": "day of the week",
        "Open": "was the store open",
        "StateHoliday": "indicates a state holiday",
        "SchoolHoliday": "affected by school holidays",
        "StoreType": "type of store",
        "Assortment": "assortment level",
        "CompetitionDistance": "distance in meters to the nearest competitor store",
        "Promo": "product is in promotion",
        "Promo2": "special promotion",
        }

        stores = pd.read_csv(os.path.join(self.current_dir,"store.csv"))
        df = (
            pd.read_csv(os.path.join(self.current_dir,"train.csv"), parse_dates=True)
            .assign(
                StateHoliday=lambda df: df.StateHoliday.map(
                    {
                        "a": "public holiday",
                        "b": "Easter holiday",
                        "c": "Christmas",
                        "0": "none",
                        0: "none",
                    }
                )
            )
            .merge(stores, how="left", on="Store")
            .sort_values(["Store", "Date"])
            .assign(
                Assortment=lambda df: df.Assortment.map(
                    {"a": "basic", "b": "extra", "c": "extended"}
                )
            )
            .assign(Open=lambda df: df.Open.map({0: "closed", 1: "open"}))
            .assign(Promo=lambda df: df.Promo.map({0: "no", 1: "yes"}))
            .assign(SchoolHoliday=lambda df: df.SchoolHoliday.map({0: "no", 1: "yes"}))
            .assign(
                Promo2=lambda df: df.Promo2.map(
                    {0: "not participating", 1: "participating"}
                )
            )
            .assign(
                StoreType=lambda df: df.StoreType.map(
                    {"a": "Type A", "b": "Type B", "c": "Type C", "d": "Type D"}
                )
            )
            .assign(
                DayOfWeek=lambda df: df.DayOfWeek.map(
                    {
                        1: "Monday",
                        2: "Tuesday",
                        3: "Wednesday",
                        4: "Thursday",
                        5: "Friday",
                        6: "Saturday",
                        7: "Sunday",
                    }
                )
            )
            .drop([c for c in stores.columns if c.startswith("Competition")], axis=1)
            .drop([c for c in stores.columns if c.startswith("Promo2S")], axis=1)
            .drop(["PromoInterval", "StoreType", "Assortment"], axis=1)
        )

        df = df[(df["Open"] == "open") & (df["Sales"] != 0)]
        df.drop(["Open"], axis=1, inplace=True)

        df["Date"] = pd.to_datetime(df["Date"])

        df["month"] = df["Date"].dt.month_name()
        df["year"] = df["Date"].dt.year
        df.drop(["Date"], axis=1, inplace=True)

        df = df.rename(columns=col_dict)
        df = df.iloc[-last_nrows:]

        train_rows = int(0.95 * last_nrows)
        df_train = df[:train_rows]
        df_val = df[train_rows:]
        df_train["target"] = np.log1p(df_train["Sales"])
        df_val['target'] = df_val['Sales']

        df_train.drop(["Sales"], axis=1, inplace=True)
        df_val.drop(["Sales"], axis=1, inplace=True)


        categorical_features = df_train.columns[
            (df_train.columns != "target") & (df_train.columns != "number of customers")
        ].tolist()
        numerical_features = ["number of customers"]

        num_max = df_train[numerical_features].abs().max()
        df_train[numerical_features] = df_train[numerical_features] / num_max
        df_val[numerical_features] = df_val[numerical_features] / num_max

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_features = len(categorical_features + numerical_features)
        self.target_column = 'target'





