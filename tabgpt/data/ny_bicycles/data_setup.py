from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import os

class NYBicyclesData(DataFrameLoader):
    def __init__(self, task_description='bicycles count'):
        super().__init__(task_description)

    def data_preparation(self, df, train=True):
        df = df.rename(
            columns={"Low Temp (°F)": "Low Temp (F)", "High Temp (°F)": "High Temp (F)"}
        )
        df = df.melt(
            id_vars=["Date", "High Temp (F)", "Low Temp (F)", "Precipitation"],
            value_vars=[
                "Brooklyn Bridge",
                "Manhattan Bridge",
                "Williamsburg Bridge",
                "Queensboro Bridge",
            ],
            var_name="bridge",
            value_name="bicycles count",
        )

        # df['weekday'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['date'] = pd.to_datetime(df['Date'])
        df["weekday"] = df['date'].dt.day_name()

        if train:
            df["target"] = np.log(1 + df["bicycles count"])
        else:
            df["target"] = df["bicycles count"]

        return df

    def setup(self):
        current_dir = self.current_dir
        df_train1 = pd.read_excel(
            os.path.join(current_dir,"2017 Monthly Bike Count Totals for East River Bridges/04 April 2017 Cyclist Numbers for Web.xlsx"),
            usecols="B,D,E,F,G,H,I,J",
            skiprows=lambda x: x in range(5) or x > 35,
            converters={"Precipitation": lambda x: 0.0 if x == "T" else x},
        )
        df_train2 = pd.read_excel(
            os.path.join(current_dir,"2017 Monthly Bike Count Totals for East River Bridges/05 May 2017 Cyclist Numbers for Web.xlsx"),
            usecols="B,D,E,F,G,H,I,J",
            skiprows=lambda x: x in range(5) or x > 36,
            converters={"Precipitation": lambda x: 0.0 if x == "T" else x},
        )
        df_train3 = pd.read_excel(
            os.path.join(current_dir,"2017 Monthly Bike Count Totals for East River Bridges/06 June 2017 Cyclist Numbers for Web.xlsx"),
            usecols="B,D,E,F,G,H,I,J",
            skiprows=lambda x: x in range(5) or x > 35,
            converters={"Precipitation": lambda x: 0.0 if x == "T" else x},
        )
        df_train4 = pd.read_excel(
            os.path.join(current_dir,"2017 Monthly Bike Count Totals for East River Bridges/07 July 2017 Cyclist Numbers for Web.xlsx"),
            usecols="B,D,E,F,G,H,I,J",
            skiprows=lambda x: x in range(5) or x > 36,
            converters={"Precipitation": lambda x: 0.0 if x == "T" else x},
        )
        df_train5 = pd.read_excel(
            os.path.join(current_dir,"2017 Monthly Bike Count Totals for East River Bridges/08 August 2017 Cyclist Numbers for Web.xlsx"),
            usecols="B,D,E,F,G,H,I,J",
            skiprows=lambda x: x in range(5) or x > 36,
            converters={"Precipitation": lambda x: 0.0 if x == "T" else x},
        )
        df_train6 = pd.read_excel(
            os.path.join(current_dir,"2017 Monthly Bike Count Totals for East River Bridges/09 September 2017 Cyclist Numbers for Web.xlsx"),
            usecols="B,D,E,F,G,H,I,J",
            skiprows=lambda x: x in range(5) or x > 35,
            converters={"Precipitation": lambda x: 0.0 if x == "T" else x},
        )
        df_train = pd.concat(
            [df_train1, df_train2, df_train3, df_train4, df_train5, df_train6]
        )

        df_test = pd.read_excel(
            os.path.join(current_dir,"2017 Monthly Bike Count Totals for East River Bridges/10 October 2017 Cyclist Numbers.xlsx"),
            usecols="B,D,E,F,G,H,I,J",
            skiprows=lambda x: x in range(5) or x > 36,
            converters={"Precipitation": lambda x: 0.0 if x == "T" else x},
        )

        categorical_features = [
            "weekday",
            "bridge",
        ]
        numerical_features = [
            "Precipitation",
            "High Temp (F)",
            "Low Temp (F)",
        ]

        df_train = self.data_preparation(df_train)
        df_val = self.data_preparation(df_test,train=False)

        num_max = df_train[numerical_features].abs().max()
        df_train[numerical_features] = df_train[numerical_features] / num_max
        df_val[numerical_features] = df_val[numerical_features] / num_max

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_features = len(numerical_features + categorical_features)
        self.target_column = "target"



if __name__ == '__main__':
    nybicycles = NYBicyclesData()
    nybicycles.setup()

    print(nybicycles.n_features)
    