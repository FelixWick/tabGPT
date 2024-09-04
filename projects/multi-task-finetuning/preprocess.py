import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from tabgpt.col_embed import get_column_embeddings, store_row_as_file
import torch



def ewma_prediction(df, group_cols, col, alpha, horizon):
    df.sort_values(["date"], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)
    df["past sales"] = df_grouped[col].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
    return df


def ewma_merge(df_test, df_train, ewma_col, group_cols):
    def get_latest_ewmas(df):
        return df.loc[df["date"] == df["date"].max(), ewma_col]

    df_train_latest_ewma = df_train[["date", ewma_col] + group_cols].groupby(group_cols).apply(get_latest_ewmas).reset_index()

    df_test = df_test.merge(df_train_latest_ewma[[ewma_col] + group_cols], on=group_cols, how="left")

    return df_test


def seasonality_features(df):
    df['date'] = pd.to_datetime(df['date'])
    # df["weekday"] = df['date'].dt.dayofweek
    df["weekday"] = df['date'].dt.day_name()
    df["day in month"] = df['date'].dt.day
    df["day in year"] = df['date'].dt.dayofyear
    return df


def get_events(df):
    for event_date in ['2015-08-07', '2016-08-12', '2017-08-11']:
        for event_days in range(0, 6):
            df.loc[df['date'] == str((pd.to_datetime(event_date) + datetime.timedelta(days=event_days))).split(" ")[0], "days around Primer Grito de Independencia"] = event_days
    return df


def get_data_store_sales():
    # use data from Kaggle competition https://www.kaggle.com/competitions/store-sales-time-series-forecasting
    df_train_full = pd.read_csv("../store_sales/train.csv")
    df_train_full = df_train_full[~np.isin(df_train_full["date"], ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"])]
    df_train_full = df_train_full[(df_train_full["date"] < "2016-04-16") | (df_train_full["date"] > "2016-05-01")]

    df_oil = pd.read_csv("../store_sales/oil.csv")
    df_train_full = df_train_full.merge(df_oil, on="date", how="left")

    df_train_full = seasonality_features(df_train_full)

    df_train_full = get_events(df_train_full)

    # take just a small data set for testing
    df_train_full = df_train_full[df_train_full["date"] >= "2016-11-01"].reset_index(drop=True)
    df_train_full = df_train_full[(df_train_full["store_nbr"].isin([1, 2, 3])) & (df_train_full["family"].isin(["LIQUOR,WINE,BEER", "EGGS", "MEATS"]))].reset_index(drop=True)

    colname_dict = {
        "store_nbr": "store",
        "family": "product group",
        "onpromotion": "items on promotion",
        "dcoilwtico": "oil price",
    }
    df_train_full.rename(columns=colname_dict, inplace=True)
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

    features = categorical_features + numerical_features

    df_train = df_train_full[df_train_full["date"] <= "2017-07-30"].reset_index(drop=True)
    df_val = df_train_full[df_train_full["date"] >= "2017-07-31"].reset_index(drop=True)

    df_train["target"] = np.log(1 + df_train["sales"])
    df_val["target"] = df_val["sales"]

    ewma_groups = ["store", "product group", "weekday"]
    df_train = ewma_prediction(df_train, ewma_groups, "target", 0.15, 1)
    df_val = ewma_merge(df_val, df_train, "past sales", ewma_groups)

    num_max = df_train[numerical_features].abs().max()
    df_train[numerical_features] = df_train[numerical_features] / num_max
    df_val[numerical_features] = df_val[numerical_features] / num_max

    return df_train, df_val, features, categorical_features, numerical_features


def get_data_house_prices():
    # use data from Kaggle competition https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
    df_train_full = pd.read_csv("../house_prices/train.csv")

    categorical_features = [
        "OverallQual",
        "ExterQual",
        "Neighborhood",
        "BsmtQual",
        "KitchenQual",
    ]
    numerical_features = [
        "GarageCars",
        "GrLivArea",
        "GarageArea",
        "TotalBsmtSF",
        "YearBuilt",         
    ]

    features = categorical_features + numerical_features

    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    df_train["target"] = np.log(1 + df_train["SalePrice"])
    df_val["target"] = df_val["SalePrice"]

    num_max = df_train[numerical_features].abs().max()
    df_train[numerical_features] = df_train[numerical_features] / num_max
    df_val[numerical_features] = df_val[numerical_features] / num_max

    return df_train, df_val, features, categorical_features, numerical_features


def get_data_bicycles_count():
    df_train1 = pd.read_excel(
        "../NY_bicycles/2017 Monthly Bike Count Totals for East River Bridges/04 April 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 35,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train2 = pd.read_excel(
        "../NY_bicycles/2017 Monthly Bike Count Totals for East River Bridges/05 May 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 36,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train3 = pd.read_excel(
        "../NY_bicycles/2017 Monthly Bike Count Totals for East River Bridges/06 June 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 35,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train4 = pd.read_excel(
        "../NY_bicycles/2017 Monthly Bike Count Totals for East River Bridges/07 July 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 36,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train5 = pd.read_excel(
        "../NY_bicycles/2017 Monthly Bike Count Totals for East River Bridges/08 August 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 36,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train6 = pd.read_excel(
        "../NY_bicycles/2017 Monthly Bike Count Totals for East River Bridges/09 September 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 35,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train = pd.concat([df_train1, df_train2, df_train3, df_train4, df_train5, df_train6])

    df_test = pd.read_excel(
        "../NY_bicycles/2017 Monthly Bike Count Totals for East River Bridges/10 October 2017 Cyclist Numbers.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 36,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )

    def data_preparation(df):
        df = df.rename(columns={"Low Temp (°F)": "Low Temp (F)", "High Temp (°F)": "High Temp (F)"})
        df = df.melt(
            id_vars=["Date", "High Temp (F)", "Low Temp (F)", "Precipitation"],
            value_vars=["Brooklyn Bridge", "Manhattan Bridge", "Williamsburg Bridge", "Queensboro Bridge"],
            var_name="bridge",
            value_name="bicycles count"
        )

        df['date'] = pd.to_datetime(df['Date'])
        df['weekday'] = df['date'].dt.day_name()

        return df

    df_train = data_preparation(df_train)
    df_val = data_preparation(df_test)

    categorical_features = [
        "weekday",
        "bridge",
    ]
    numerical_features = [
        "Precipitation",
        "High Temp (F)",
        "Low Temp (F)",
    ]

    features = categorical_features + numerical_features

    df_train["target"] = np.log(1 + df_train["bicycles count"])
    df_val["target"] = df_val["bicycles count"]

    num_max = df_train[numerical_features].abs().max()
    df_train[numerical_features] = df_train[numerical_features] / num_max
    df_val[numerical_features] = df_val[numerical_features] / num_max

    return df_train, df_val, features, categorical_features, numerical_features


def get_data_simulated_demand():
    df_train = pd.read_parquet("../demand-forecasting-simulated/train.parquet.gzip")
    df_test = pd.read_parquet("../demand-forecasting-simulated/test.parquet.gzip")
    df_test_results = pd.read_parquet("../demand-forecasting-simulated/test_results.parquet.gzip")
    df_test = df_test.merge(df_test_results, on=['P_ID', 'L_ID', 'DATE'])

    # take just a small data set for testing
    df_train = df_train[df_train["DATE"] >= "2021-10-01"].reset_index(drop=True)

    def data_preparation(df):
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

    df_train = data_preparation(df_train)
    df_test = data_preparation(df_test)

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

    features = categorical_features + numerical_features

    num_max = df_train[numerical_features].abs().max()
    df_train[numerical_features] = df_train[numerical_features] / num_max
    df_test[numerical_features] = df_test[numerical_features] / num_max

    return df_train, df_test, features, categorical_features, numerical_features


if __name__ == '__main__':
    max_features = 10
    # STORE SALES
    (
        df_train,
        df_val,
        features,
        categorical_features,
        numerical_features,
    ) = get_data_store_sales()
    features_embeds = get_column_embeddings(
        df_train,
        "store sales",
        categorical_features,
        numerical_features,
        number_of_cols=max_features,
    ).to(torch.float16)
    store_row_as_file(
        store_dir="./data/store-sales",
        dataset_name="store-sales",
        features_embeds=features_embeds,
        targets=df_train["target"].to_numpy(),
    )

    # HOUSE PRICES
    (
        df_train,
        df_val,
        features,
        categorical_features,
        numerical_features,
    ) = get_data_house_prices()
    features_embeds = get_column_embeddings(
        df_train,
        "house prices",
        categorical_features,
        numerical_features,
        number_of_cols=max_features,
    ).to(torch.float16)
    store_row_as_file(
        store_dir="./data/house-prices",
        dataset_name="house-prices",
        features_embeds=features_embeds,
        targets=df_train["target"].to_numpy(),
    )

    # SIMULATED DEMAND
    (
        df_train,
        df_val,
        features,
        categorical_features,
        numerical_features,
    ) = get_data_simulated_demand()
    features_embeds = get_column_embeddings(
        df_train,
        "retail demand forecasting",
        categorical_features,
        numerical_features,
        number_of_cols=max_features,
    ).to(torch.float16)
    store_row_as_file(
        store_dir="./data/demand-forecasting",
        dataset_name="demand-forecasting",
        features_embeds=features_embeds,
        targets=df_train["target"].to_numpy(),
    )

    # NY Bicycles COUNT
    (
        df_train,
        df_val,
        features,
        categorical_features,
        numerical_features,
    ) = get_data_bicycles_count()
    features_embeds = get_column_embeddings(
        df_train,
        "bicycles count",
        categorical_features,
        numerical_features,
        number_of_cols=max_features,
    ).to(torch.float16)
    store_row_as_file(
        store_dir="./data/ny-bicycles",
        dataset_name="ny-bicycles",
        features_embeds=features_embeds,
        targets=df_train["target"].to_numpy(),
    )