import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
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
        ts = df.groupby(['date'])[['sales', 'yhat']].sum().reset_index()
    else:
        ts = df.groupby(['date'])['sales'].sum().reset_index()
    plt.figure()
    ts.index = ts['date']
    ts['sales'].plot(style='r', label="sales")
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


def main(test):
    np.random.seed(666)
    torch.manual_seed(42)

    # use data from Kaggle competition https://www.kaggle.com/competitions/store-sales-time-series-forecasting
    df_train_full = pd.read_csv("train.csv")
    df_train_full = df_train_full[~np.isin(df_train_full["date"], ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"])]
    df_train_full = df_train_full[(df_train_full["date"] < "2016-04-16") | (df_train_full["date"] > "2016-05-01")]

    df_oil = pd.read_csv("oil.csv")
    df_train_full = df_train_full.merge(df_oil, on="date", how="left")

    df_train_full = seasonality_features(df_train_full)

    df_train_full = get_events(df_train_full)

    df_train_full["sales_transformed"] = np.log(1 + df_train_full["sales"])

    # take just a small data set for testing
    # used single training of concept paper:
    # df_train_full = df_train_full[df_train_full["date"] >= "2017-05-01"].reset_index(drop=True)
    # used in individual comparison for cross-training of concept paper:
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

    if test:
        df_train = df_train_full
        df_test = pd.read_csv("test.csv")
        df_test = df_test.merge(df_oil, on="date", how="left")
        df_test = seasonality_features(df_test)
        df_test = get_events(df_test)
        df_test = df_test[(df_test["store_nbr"].isin([1, 2, 3])) & (df_test["family"].isin(["LIQUOR,WINE,BEER", "EGGS", "MEATS"]))].reset_index()
        df_test.rename(columns=colname_dict, inplace=True)
    else:
        df_train = df_train_full[df_train_full["date"] <= "2017-07-30"].reset_index(drop=True)
        df_test = df_train_full[df_train_full["date"] >= "2017-07-31"].reset_index(drop=True)

    ewma_groups = ["store", "product group", "weekday"]
    df_train = ewma_prediction(df_train, ewma_groups, "sales_transformed", 0.15, 1)
    df_test = ewma_merge(df_test, df_train, "past sales", ewma_groups)

    num_max = df_train[numerical_features].abs().max()
    df_train[numerical_features] = df_train[numerical_features] / num_max
    df_test[numerical_features] = df_test[numerical_features] / num_max

    features_embeds_train = get_column_embeddings(df_train, "store sales", categorical_features, numerical_features, number_of_cols=len(features))

    train_dataset = TensorDataset(
        features_embeds_train,
        torch.tensor(df_train["sales_transformed"].tolist(), dtype=torch.float32)
    )

    max_length = len(features) + 1

    # tabGPT model
    model_config = tabGPT.get_default_config()
    model_config.model_type = 'gpt-micro'
    model_config.vocab_size = 50257 # openai's model vocabulary
    model_config.block_size = max_length # 1024 is openai's model block_size
    model_config.n_output_nodes = 1
    model = tabGPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 1000000
    train_config.epochs = 65
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
    df_train = predict(model, DataLoader(train_dataset, batch_size=32), df_train)
    evaluation(df_train["sales"], df_train["yhat"])
    plot_timeseries(df_train, "train", True)
    for pg in df_train["product group"].unique():
        if pg != "BREAD/BAKERY":
            plot_timeseries(df_train[df_train["product group"] == pg], pg + "_train", True)
        else:
            plot_timeseries(df_train[df_train["product group"] == pg], "BREAD_BAKERY_train", True)

    features_embeds_test = get_column_embeddings(df_test, "store sales", categorical_features, numerical_features, number_of_cols=len(features))

    if test:
        test_dataset = TensorDataset(
            features_embeds_test,
            torch.tensor(df_test["store"].tolist(), dtype=torch.float32)
        )
    else:
        test_dataset = TensorDataset(
            features_embeds_test,
            torch.tensor(df_test["sales_transformed"].tolist(), dtype=torch.float32)
        )

    df_test = predict(model, DataLoader(test_dataset, batch_size=32), df_test)
    if test:
        pd.concat([df_test["id"], df_test["yhat"]], axis=1).rename(columns={"yhat": "sales"}).to_csv("submission.csv", index=False)
    else:
        evaluation(df_test["sales"], df_test["yhat"])
        plot_timeseries(df_test, "val", True)
        for pg in df_test["product group"].unique():
            if pg != "BREAD/BAKERY":
                plot_timeseries(df_test[df_test["product group"] == pg], pg + "_val", True)
            else:
                plot_timeseries(df_test[df_test["product group"] == pg], "BREAD_BAKERY_val", True)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args.test)
