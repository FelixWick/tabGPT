from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from tabgpt.utils import stratify


class RossmannData(DataFrameLoader):
    def __init__(self, task_description='Rossmann store products'):
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

        # Store -> unique store Id
        df = stratify(df,groupby_cols=['Store'])
        df = df.rename(columns=col_dict)

        df_train, df_val = train_test_split(df,test_size=0.1,stratify=df[['unique store Id']])

        categorical_features = df_train.columns[
            (df_train.columns != "Sales") & (df_train.columns != "number of customers")
        ].tolist()
        numerical_features = ["Sales", "number of customers"]

        self.setup_scaler(numerical_features)
        self.scale_columns(df_train, mode='train')
        self.scale_columns(df_val)

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.set_target_column(main_target='Sales', additional_ones=True)





