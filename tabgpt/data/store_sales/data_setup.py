
from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import datetime
import os

class StoreSalesData(DataFrameLoader):
    def __init__(self, task_description='store sales'):
        super().__init__(task_description)
        self.colname_dict = {
            "store_nbr": "store",
            "family": "product group",
            "onpromotion": "items on promotion",
            "dcoilwtico": "oil price",
        }


    def setup(self):
        # use data from Kaggle competition https://www.kaggle.com/competitions/store-sales-time-series-forecasting
        df_train_full = pd.read_csv(os.path.join(self.current_dir,"train.csv"))
        df_train_full = df_train_full[~np.isin(df_train_full["date"], ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"])]
        df_train_full = df_train_full[(df_train_full["date"] < "2016-04-16") | (df_train_full["date"] > "2016-05-01")]

        df_oil = pd.read_csv(os.path.join(self.current_dir,"oil.csv"))
        df_train_full = df_train_full.merge(df_oil, on="date", how="left")

        df_train_full = self.seasonality_features(df_train_full)

        df_train_full = self.get_events(df_train_full)

        # take just a small data set for testing
        df_train_full = df_train_full[df_train_full["date"] >= "2016-11-01"].reset_index(drop=True)
        df_train_full = df_train_full[(df_train_full["store_nbr"].isin([1, 2, 3])) & (df_train_full["family"].isin(["LIQUOR,WINE,BEER", "EGGS", "MEATS"]))].reset_index(drop=True)


        df_train_full.rename(columns=self.colname_dict, inplace=True)
        categorical_features = [
            "store",
            "product group",
            "weekday",
        ]
        numerical_features = [
            "items on promotion",
            "oil price",
            "day in month",
            "day in year",
            "days around Primer Grito de Independencia",
            "past sales",
        ]
        
        df_train = df_train_full[df_train_full["date"] <= "2017-07-30"].reset_index(drop=True)
        df_val = df_train_full[df_train_full["date"] >= "2017-07-31"].reset_index(drop=True)

        df_train["target"] = np.log(1 + df_train["sales"])
        df_val["target"] = df_val["sales"]

        ewma_groups = ["store", "product group", "weekday"]
        df_train = self.ewma_prediction(df_train, ewma_groups, "target", 0.15, 1)
        df_val = self.ewma_merge(df_val, df_train, "past sales", ewma_groups)

        num_max = df_train[numerical_features].abs().max()
        df_train[numerical_features] = df_train[numerical_features] / num_max
        df_val[numerical_features] = df_val[numerical_features] / num_max

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_features = len(numerical_features + categorical_features)
        self.target_column = "target"

    def test_setup(self):
        df_oil = pd.read_csv(os.path.join(self.current_dir,"oil.csv"))

        df_test = pd.read_csv(os.path.join(self.current_dir,"test.csv"))
        df_test = df_test.merge(df_oil, on="date", how="left")
        df_test = self.seasonality_features(df_test)
        df_test = self.get_events(df_test)
        df_test.rename(columns=self.colname_dict, inplace=True)
        self.df_test = df_test
        self.numerical_features.remove('past sales')
        self.n_features -= 1
        self.target_column = 'store'
    
    def ewma_prediction(self, df, group_cols, col, alpha, horizon):
        df.sort_values(["date"], inplace=True)
        df_grouped = df.groupby(group_cols, group_keys=False)
        df["past sales"] = df_grouped[col].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
        return df


    def ewma_merge(self, df_test, df_train, ewma_col, group_cols):
        def get_latest_ewmas(df):
            return df.loc[df["date"] == df["date"].max(), ewma_col]

        df_train_latest_ewma = df_train[["date", ewma_col] + group_cols].groupby(group_cols).apply(get_latest_ewmas).reset_index()

        df_test = df_test.merge(df_train_latest_ewma[[ewma_col] + group_cols], on=group_cols, how="left")

        return df_test


    def seasonality_features(self, df):
        df['date'] = pd.to_datetime(df['date'])
        # df["weekday"] = df['date'].dt.dayofweek
        df["weekday"] = df['date'].dt.day_name()
        df["day in month"] = df['date'].dt.day
        df["day in year"] = df['date'].dt.dayofyear
        return df


    def get_events(self,df):
        for event_date in ['2015-08-07', '2016-08-12', '2017-08-11']:
            for event_days in range(0, 6):
                df.loc[df['date'] == str((pd.to_datetime(event_date) + datetime.timedelta(days=event_days))).split(" ")[0], "days around Primer Grito de Independencia"] = event_days
        return df




if __name__ == '__main__':
    storesales = StoreSalesData()
    storesales.setup()

    print(storesales.n_features)