import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader

from tabgpt.model import tabGPT
from tabgpt.trainer import Trainer
from tabgpt.col_embed import get_column_embeddings

from IPython import embed


if torch.cuda.is_available():       
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def plot_timeseries(df, suffix, include_preds=False):
    if include_preds:
        ts = df.groupby(['date'])[['target', 'yhat']].sum().reset_index()
    else:
        ts = df.groupby(['date'])['target'].sum().reset_index()
    plt.figure()
    ts.index = ts['date']
    ts['target'].plot(style='r', label="target")
    if include_preds:
        ts['yhat'].plot(style='b-.', label="predictions")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig("ts_{}.png".format(suffix))
    plt.clf()


def evaluation(y, yhat):
    print('RMSLE: ', root_mean_squared_log_error(y, yhat))
    print('mean(y): ', np.mean(y))


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


def predict(model, dataloader, df):
    model.eval()

    yhat = []
    for input_ids, _ in dataloader:
        with torch.no_grad():
            yhat += model.generate(input_ids.to(device)).cpu().detach().numpy().tolist()

    df["yhat"] = yhat
    df["yhat"] = np.clip(df["yhat"], 0, None)
    df["yhat"] = np.exp(df["yhat"]) - 1
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


def main(pretrained):
    np.random.seed(666)
    torch.manual_seed(42)

    df_train_store_sales, df_val_store_sales, features_store_sales, categorical_features_store_sales, numerical_features_store_sales = get_data_store_sales()
    df_train_house_prices, df_val_house_prices, features_house_prices, categorical_features_house_prices, numerical_features_house_prices = get_data_house_prices()
    df_train_bicycles_count, df_val_bicycles_count, features_bicycles_count, categorical_features_bicycles_count, numerical_features_bicycles_count = get_data_bicycles_count()
    df_train_simulated_demand, df_val_simulated_demand, features_simulated_demand, categorical_features_simulated_demand, numerical_features_simulated_demand = get_data_simulated_demand()

    print("train samples store sales: ", len(df_train_store_sales))
    print("train samples house prices: ", len(df_train_house_prices))
    print("train samples bicycles count: ", len(df_train_bicycles_count))
    print("train samples simulated demand: ", len(df_train_simulated_demand))

    max_features = max(len(features_store_sales), len(features_house_prices), len(features_bicycles_count), len(features_simulated_demand))

    features_embeds_train_store_sales = get_column_embeddings(df_train_store_sales, "store sales", categorical_features_store_sales, numerical_features_store_sales, number_of_cols=max_features)
    features_embeds_train_house_prices = get_column_embeddings(df_train_house_prices, "house prices", categorical_features_house_prices, numerical_features_house_prices, number_of_cols=max_features)
    features_embeds_train_bicycles_count = get_column_embeddings(df_train_bicycles_count, "bicycles count", categorical_features_bicycles_count, numerical_features_bicycles_count, number_of_cols=max_features)
    features_embeds_train_simulated_demand = get_column_embeddings(df_train_simulated_demand, "retail demand forecasting", categorical_features_simulated_demand, numerical_features_simulated_demand, number_of_cols=max_features)

    features_embeds_train = torch.cat((features_embeds_train_store_sales, features_embeds_train_house_prices, features_embeds_train_bicycles_count, features_embeds_train_simulated_demand), dim=0)

    targets_train = df_train_store_sales["target"].tolist() + df_train_house_prices["target"].tolist() + df_train_bicycles_count["target"].tolist() + df_train_simulated_demand["target"].tolist()

    train_dataset = TensorDataset(
        features_embeds_train, 
        torch.tensor(targets_train, dtype=torch.float32)
        )

    # tabGPT model
    if pretrained:
        model = tabGPT.from_pretrained('gpt2', 1)
    else:
        model_config = tabGPT.get_default_config()
        model_config.model_type = 'gpt-micro'
        model_config.vocab_size = 50257 # openai's model vocabulary
        model_config.block_size = max_features + 1 # 1024 is openai's model block_size
        model_config.n_output_nodes = 1
        model = tabGPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 100000
    train_config.epochs = 80
    train_config.num_workers = 0
    train_config.batch_size = 64
    train_config.observe_train_loss = True
    trainer = Trainer(train_config, model, train_dataset)

    if train_config.observe_train_loss:
        def epoch_end_callback(trainer):
            print(f"epoch {trainer.epoch}: train loss {np.sqrt(trainer.aggregated_loss.detach().cpu())}")
        trainer.set_callback('on_epoch_end', epoch_end_callback)
    else:
        def batch_end_callback(trainer):
            if trainer.iter_num % 100 == 0:
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()

    # inference
    features_embeds_val_store_sales = get_column_embeddings(df_val_store_sales, "store sales", categorical_features_store_sales, numerical_features_store_sales, number_of_cols=max_features)
    features_embeds_val_house_prices = get_column_embeddings(df_val_house_prices, "house prices", categorical_features_house_prices, numerical_features_house_prices, number_of_cols=max_features)
    features_embeds_val_bicycles_count = get_column_embeddings(df_val_bicycles_count, "bicycles count", categorical_features_bicycles_count, numerical_features_bicycles_count, number_of_cols=max_features)
    features_embeds_val_simulated_demand = get_column_embeddings(df_val_simulated_demand, "retail demand forecasting", categorical_features_simulated_demand, numerical_features_simulated_demand, number_of_cols=max_features)

    val_dataset_store_sales = TensorDataset(
        features_embeds_val_store_sales, 
        torch.tensor(df_val_store_sales["target"].tolist(), dtype=torch.float32)
        )

    val_dataset_house_prices = TensorDataset(
        features_embeds_val_house_prices, 
        torch.tensor(df_val_house_prices["target"].tolist(), dtype=torch.float32)
        )

    val_dataset_bicycles_count = TensorDataset(
        features_embeds_val_bicycles_count, 
        torch.tensor(df_val_bicycles_count["target"].tolist(), dtype=torch.float32)
        )

    val_dataset_simulated_demand = TensorDataset(
        features_embeds_val_simulated_demand, 
        torch.tensor(df_val_simulated_demand["target"].tolist(), dtype=torch.float32)
        )

    df_val_store_sales = predict(model, DataLoader(val_dataset_store_sales, batch_size=32), df_val_store_sales)
    evaluation(df_val_store_sales["target"], df_val_store_sales["yhat"])
    plot_timeseries(df_val_store_sales, "store_sales", True)

    df_val_house_prices = predict(model, DataLoader(val_dataset_house_prices, batch_size=32), df_val_house_prices)
    evaluation(df_val_house_prices["target"], df_val_house_prices["yhat"])

    df_val_bicycles_count = predict(model, DataLoader(val_dataset_bicycles_count, batch_size=32), df_val_bicycles_count)
    evaluation(df_val_bicycles_count["target"], df_val_bicycles_count["yhat"])
    plot_timeseries(df_val_bicycles_count, "bicycles_count", True)

    df_val_simulated_demand = predict(model, DataLoader(val_dataset_simulated_demand, batch_size=32), df_val_simulated_demand)
    evaluation(df_val_simulated_demand["target"], df_val_simulated_demand["yhat"])
    plot_timeseries(df_val_simulated_demand, "simulated_demand", True)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    main(args.pretrained)
