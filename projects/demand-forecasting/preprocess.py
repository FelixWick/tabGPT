import numpy as np
import pandas as pd
from IPython import embed
import torch
from tabgpt.col_embed import get_column_embeddings, store_row_as_file



def preprocess_favorita(path, n_rows=100_000):

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
        f"{path}/train.csv/train.csv", parse_dates=["date"], skiprows=range(1, 124600000)
    )  # result ~ 800.000 rows
    items = pd.read_csv(f"{path}/items.csv/items.csv")
    df_holiday_events = pd.read_csv(f"{path}/holidays_events.csv/holidays_events.csv")
    df_holiday_events["date"] = pd.to_datetime(df_holiday_events["date"])
    df = df_train.merge(items, how="left", on="item_nbr")
    df = df.merge(df_holiday_events, how="left", on="date")
    df["month"] = df["date"].dt.month_name(locale="English")
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.day_name()
    df.drop(["date"], axis=1, inplace=True)

    df = df[df["unit_sales"] >= 0]

    df.drop(["id"], axis=1, inplace=True)

    df = df.assign(onpromotion=lambda df: df.onpromotion.map({False: "no", True: "yes"})).assign(
        perishable=lambda df: df.perishable.map({0: "no", 1: "yes"})
    )

    df["Sales"] = np.log1p(df["unit_sales"])
    df.drop(["unit_sales"], axis=1, inplace=True)

    df = df.rename(columns=col_dict)

    df = df.iloc[-n_rows:] # last
    train_rows = int(0.95*n_rows)
    df_train = df[:train_rows]
    df_val = df[train_rows:]

    categorical_features = df.columns[df.columns != 'Sales'].tolist()
    numerical_features = []
    features = categorical_features + numerical_features

    return df_train, df_val, features, categorical_features, numerical_features



def preprocess_rossmann(path, n_rows=100_000):


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


    stores = pd.read_csv(f"{path}/store.csv")
    df = (
        pd.read_csv(f"{path}/train.csv", parse_dates=True)
        .assign(
            StateHoliday=lambda df: df.StateHoliday.map({
                "a": "public holiday",
                "b": "Easter holiday",
                "c": "Christmas",
                "0": "none",
                0: "none",
            })
        )
        .merge(stores, how="left", on="Store")
        .sort_values(["Store", "Date"])
        .assign(Assortment=lambda df: df.Assortment.map({"a": "basic", "b": "extra", "c": "extended"}))
        .assign(Open=lambda df: df.Open.map({0: "closed", 1: "open"}))
        .assign(Promo=lambda df: df.Promo.map({0: "no", 1: "yes"}))
        .assign(SchoolHoliday=lambda df: df.SchoolHoliday.map({0: "no", 1: "yes"}))
        .assign(Promo2=lambda df: df.Promo2.map({0: "not participating", 1: "participating"}))
        .assign(StoreType=lambda df: df.StoreType.map({"a": "Type A", "b": "Type B", "c": "Type C", "d": "Type D"}))
        .assign(
            DayOfWeek=lambda df: df.DayOfWeek.map({
                1: "Monday",
                2: "Tuesday",
                3: "Wednesday",
                4: "Thursday",
                5: "Friday",
                6: "Saturday",
                7: "Sunday",
            })
        )
        .drop([c for c in stores.columns if c.startswith("Competition")], axis=1)
        .drop([c for c in stores.columns if c.startswith("Promo2S")], axis=1)
        .drop(["PromoInterval", "StoreType", "Assortment"], axis=1)
    )

    df = df[(df["Open"] == "open") & (df["Sales"] != 0)]
    df.drop(["Open"], axis=1, inplace=True)

    df["Sales"] = np.log1p(df["Sales"])
    df["Date"] = pd.to_datetime(df["Date"])

    df["month"] = df["Date"].dt.month_name(locale="English")
    df["year"] = df["Date"].dt.year
    df.drop(["Date"], axis=1, inplace=True)

    df = df.rename(columns=col_dict)
    df = df.iloc[-n_rows:]

    train_rows = int(0.95*n_rows)
    df_train = df[:train_rows]
    df_val = df[train_rows:]

    categorical_features = df.columns[(df.columns != 'Sales') & (df.columns != 'number of customers')].tolist()
    numerical_features = ['number of customers']
    features = categorical_features + numerical_features

    num_max = df_train[numerical_features].abs().max()
    df_train[numerical_features] = df_train[numerical_features] / num_max
    df_val[numerical_features] = df_val[numerical_features] / num_max

    return df_train, df_val, features, categorical_features, numerical_features



def preprocess_walmart(path,n_rows=100_000):   

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

    dataset = pd.read_csv(f"{path}/train.csv/train.csv")
    features = pd.read_csv(f"{path}/features.csv/features.csv")
    stores = pd.read_csv(f"{path}/stores.csv")
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
        ["Date", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "Fuel_Price"], axis=1, inplace=True
    )



    df = df.assign(IsHoliday=lambda df: df.IsHoliday.map({False: "no", True: "yes"}))
    df = df[df["Weekly_Sales"] > 0]
    df["Weekly_Sales"] = np.log1p(df["Weekly_Sales"])
    df = df.rename(columns=col_dict)

    df = df.iloc[-n_rows:]

    train_rows = int(0.95*n_rows)
    df_train = df[:train_rows]
    df_val = df[train_rows:]

    
    numerical_features =["average temperature (in Fahrenheit)", "unemployment rate", "store size (in square feet)"]
    categorical_features =  df.columns[(df.columns != 'Weekly_Sales') & ~(df.columns.isin(numerical_features))].tolist()
    features = categorical_features + numerical_features

    num_max = df_train[numerical_features].abs().max()
    df_train[numerical_features] = df_train[numerical_features] / num_max
    df_val[numerical_features] = df_val[numerical_features] / num_max

    return df_train, df_val, features, categorical_features, numerical_features


if __name__ == '__main__':
    df_train, df_val, features, categorical_features, numerical_features = preprocess_favorita("C:/Users/70Q1985/Downloads/favorita-grocery-sales-forecasting/")
    features_embeds_favorita = get_column_embeddings(df_train, "Favorita daily sales", categorical_features, numerical_features, number_of_cols=15).to(torch.float16)
    store_row_as_file(store_dir='C:/Users/70Q1985/tabGPT/projects/demand-forecasting/data',dataset_name='Favorita',features_embeds=features_embeds_favorita,targets=df_train['Sales'].to_numpy())

    df_train, df_val, features, categorical_features, numerical_features = preprocess_rossmann("C:/Users/70Q1985/Downloads/rossmann")
    features_embeds_rossmann = get_column_embeddings(df_train, "Rossmann daily sales", categorical_features, numerical_features, number_of_cols=15).to(torch.float16)
    store_row_as_file(store_dir='C:/Users/70Q1985/tabGPT/projects/demand-forecasting/data',dataset_name='Rossmann',features_embeds=features_embeds_rossmann,targets=df_train['Sales'].to_numpy())

    df_train, df_val, features, categorical_features, numerical_features = preprocess_walmart("C:/Users/70Q1985/Downloads/walmart-recruiting-store-sales-forecasting")
    features_embeds_walmart = get_column_embeddings(df_train, "Walmart weekly sales", categorical_features, numerical_features, number_of_cols=15).to(torch.float16)
    store_row_as_file(store_dir='C:/Users/70Q1985/tabGPT/projects/demand-forecasting/data',dataset_name='Walmart',features_embeds=features_embeds_walmart,targets=df_train['Weekly_Sales'].to_numpy())