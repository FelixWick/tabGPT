import sys

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

from tabgpt.model import tabGPT
from tabgpt.trainer import Trainer
from tabgpt.col_embed import get_column_embeddings

from transformers import GPT2ForSequenceClassification

from IPython import embed


if torch.cuda.is_available():       
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def plot_timeseries(df, suffix, include_preds=False):
    if include_preds:
        ts = df.groupby(['Date'])[['bicycles count', 'yhat']].sum().reset_index()
    else:
        ts = df.groupby(['Date'])['bicycles count'].sum().reset_index()
    plt.figure()
    ts.index = ts['Date']
    ts['bicycles count'].plot(style='r', label="bicycles count")
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


def predict(model, dataloader, df):
    model.eval()

    yhat = []
    for input_ids, _ in dataloader:
        with torch.no_grad():
            yhat += model(inputs_embeds=input_ids.to(device)).logits.squeeze().cpu().detach().numpy().tolist()

    df["yhat"] = yhat
    df["yhat"] = np.clip(df["yhat"], 0, None)
    df["yhat"] = np.exp(df["yhat"]) - 1
    return df


def data_preparation(df):
    df = df.rename(columns={"Low Temp (°F)": "Low Temp (F)", "High Temp (°F)": "High Temp (F)"})
    df = df.melt(
        id_vars=["Date", "High Temp (F)", "Low Temp (F)", "Precipitation"],
        value_vars=["Brooklyn Bridge", "Manhattan Bridge", "Williamsburg Bridge", "Queensboro Bridge"],
        var_name="bridge",
        value_name="bicycles count"
    )

    # df['weekday'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['weekday'] = pd.to_datetime(df['Date']).dt.day_name()

    df["bicycles_count_transformed"] = np.log(1 + df["bicycles count"])

    return df


def get_data():
    df_train1 = pd.read_excel(
        "2017 Monthly Bike Count Totals for East River Bridges/04 April 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 35,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train2 = pd.read_excel(
        "2017 Monthly Bike Count Totals for East River Bridges/05 May 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 36,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train3 = pd.read_excel(
        "2017 Monthly Bike Count Totals for East River Bridges/06 June 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 35,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train4 = pd.read_excel(
        "2017 Monthly Bike Count Totals for East River Bridges/07 July 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 36,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train5 = pd.read_excel(
        "2017 Monthly Bike Count Totals for East River Bridges/08 August 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 36,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train6 = pd.read_excel(
        "2017 Monthly Bike Count Totals for East River Bridges/09 September 2017 Cyclist Numbers for Web.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 35,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )
    df_train = pd.concat([df_train1, df_train2, df_train3, df_train4, df_train5, df_train6])

    df_test = pd.read_excel(
        "2017 Monthly Bike Count Totals for East River Bridges/10 October 2017 Cyclist Numbers.xlsx",
        usecols="B,D,E,F,G,H,I,J",
        skiprows=lambda x: x in range(5) or x > 36,
        converters={"Precipitation": lambda x: 0.0 if x == "T" else x}
    )

    df_train = data_preparation(df_train)
    df_test = data_preparation(df_test)

    return df_train, df_test


def main(args):
    np.random.seed(666)
    torch.manual_seed(42)

    # use data from https://data.cityofnewyork.us/Transportation/Bicycle-Counts-for-East-River-Bridges-Historical-/gua4-p9wg/about_data
    df_train, df_val = get_data()

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

    num_max = df_train[numerical_features].abs().max()
    df_train[numerical_features] = df_train[numerical_features] / num_max
    df_val[numerical_features] = df_val[numerical_features] / num_max

    features_embeds_train = get_column_embeddings(df_train, "bicycles count", categorical_features, numerical_features, number_of_cols=len(features))
    features_embeds_val = get_column_embeddings(df_val, "bicycles count", categorical_features, numerical_features, number_of_cols=len(features))

    max_length = len(features) + 1

    train_dataset = TensorDataset(
        features_embeds_train, 
        torch.tensor(df_train["bicycles_count_transformed"].tolist(), dtype=torch.float32)
        )

    val_dataset = TensorDataset(
        features_embeds_val, 
        torch.tensor(df_val["bicycles_count_transformed"].tolist(), dtype=torch.float32)
        )

    # tabGPT model
    model_config = tabGPT.get_default_config()
    model_config.model_type = 'gpt-micro'
    model_config.vocab_size = 50257 # openai's model vocabulary
    model_config.block_size = max_length # 1024 is openai's model block_size
    model_config.n_output_nodes = 1
    # model = tabGPT(model_config)
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=1)
    model.config.pad_token_id = 0

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 100000
    train_config.epochs = 200 # used in individual comparison for cross-training of concept paper
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
    evaluation(df_train["bicycles count"], df_train["yhat"])
    plot_timeseries(df_train, "train", True)
    for pg in df_train["bridge"].unique():
        plot_timeseries(df_train[df_train["bridge"] == pg], pg + "_train", True)

    df_val = predict(model, DataLoader(val_dataset, batch_size=32), df_val)
    evaluation(df_val["bicycles count"], df_val["yhat"])
    plot_timeseries(df_val, "val", True)
    for pg in df_val["bridge"].unique():
        plot_timeseries(df_val[df_val["bridge"] == pg], pg + "_val", True)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
