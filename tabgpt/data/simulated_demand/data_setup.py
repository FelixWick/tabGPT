from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

class SimulatedDemandData(DataFrameLoader):
    def __init__(self, task_description='retail demand forecasting'):
        super().__init__(task_description)
        self.df_test_results = None


    def setup(self, low=True):
        current_dir = self.current_dir      

        if low:
            df_train = pd.read_parquet(os.path.join(current_dir,"train.parquet.gzip"))
            df_test = pd.read_parquet(os.path.join(current_dir,"test.parquet.gzip"))
            df_test_results = pd.read_parquet(os.path.join(current_dir,"test_results.parquet.gzip"))
            df_test = df_test.merge(df_test_results, on=['P_ID', 'L_ID', 'DATE'])


        else: # high
            df_train = pd.read_parquet(os.path.join(current_dir,"train_high.parquet.gzip"))
            df_test = pd.read_parquet(os.path.join(current_dir,"test_high.parquet.gzip"))
            df_test_results = pd.read_parquet(os.path.join(current_dir,"test_results_high.parquet.gzip"))
            df_test = df_test.merge(df_test_results, on=['P_ID', 'L_ID', 'DATE'])



        # take just a small data set for testing
        df_train = df_train[df_train["DATE"] >= "2021-10-01"].reset_index(drop=True)

        df_train = self.data_preparation(df_train)
        df_test = self.data_preparation(df_test)

        df_train["target"] = np.log(1 + df_train["SALES"])
        df_test["target"] = df_test["SALES"]

        # ewma_groups = ["location id", "product id", "weekday"]
        # df_train = ewma_prediction(df_train, ewma_groups, "target", 0.15, 1)
        # df_test = ewma_merge(df_test, df_train, "past sales", ewma_groups)

        categorical_features = [
            "product id",
            "product group id",
            "location id",
            "type of promotion",
            "weekday",
        ]
        numerical_features = [
            "normal price",
            "sales area", 
            "sales price",
            "day in month",
            "day in year",
            # "past sales",
        ]


        num_max = df_train[numerical_features].abs().max()
        df_train[numerical_features] = df_train[numerical_features] / num_max
        df_test[numerical_features] = df_test[numerical_features] / num_max


        self.df_train = df_train
        self.df_test = df_test
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_features = len(numerical_features + categorical_features)
        self.target_column = "target"
        logging.info(f'Training set stored with {len(self.df_train)} rows')
        logging.info(f'{self.n_features} features used for prediction')

    def data_preparation(self, df):
        df.rename(
            columns={
                "P_ID": "product id",
                "PG_ID_3": "product group id",
                "NORMAL_PRICE": "normal price",
                "L_ID": "location id",
                "SALES_AREA": "sales area",
                "PROMOTION_TYPE": "type of promotion",
                "SALES_PRICE": "sales price",
            },
            inplace=True,
        )

        df["date"] = pd.to_datetime(df["DATE"])
        df["weekday"] = df['date'].dt.day_name()
        df["day in month"] = df['date'].dt.day
        df["day in year"] = df['date'].dt.dayofyear

        return df



if __name__ == '__main__':
    simulateddemand = SimulatedDemandData()
    simulateddemand.setup()

    print(simulateddemand.n_features)