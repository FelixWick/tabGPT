import os
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
from transformers import Trainer as HF_Trainer
from transformers import TrainingArguments


from sklearn.metrics import root_mean_squared_log_error
import itertools
import datetime
from peft import get_peft_model, LoraConfig, TaskType

import lightning as L
import pandas as pd
from tabgpt.model import tabGPT
from tabgpt.trainer import Trainer
from tabgpt.col_embed import get_column_embeddings
from pl_model import TabGPT

from IPython import embed

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


class TabularDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = os.listdir(folder_path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])

        data = torch.tensor(np.load(file_path))

        # Extract label from the file name or other source
        dataset_idx, target = self.extract_info_from_filename(self.file_names[idx])

        return data, target

    def extract_info_from_filename(self, filename):
        dataset_name, target, _ = filename.split("_")
        dataset_idx = self.dataset_name_to_idx(dataset_name)

        return dataset_idx, torch.tensor(float(target), dtype=torch.float16)

    def dataset_name_to_idx(self, dataset_name):
        if dataset_name == "Favorita":
            return torch.tensor([0], dtype=torch.int8)
        if dataset_name == "Rossmann":
            return torch.tensor([1], dtype=torch.int8)
        if dataset_name == "Walmart":
            return torch.tensor([2], dtype=torch.int8)


def plot_timeseries(df, suffix, include_preds=False):
    if include_preds:
        ts = df.groupby(["date"])[["sales", "yhat"]].sum().reset_index()
    else:
        ts = df.groupby(["date"])["sales"].sum().reset_index()
    plt.figure()
    ts.index = ts["date"]
    ts["sales"].plot(style="r", label="sales")
    if include_preds:
        ts["yhat"].plot(style="b-.", label="predictions")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig("ts_{}.png".format(suffix))
    plt.clf()
    plt.close()


def evaluation(y, yhat):
    print("RMSLE: ", root_mean_squared_log_error(y, yhat))
    print("mean(y): ", np.mean(y))


def ewma_prediction(df, group_cols, col, alpha, horizon):
    df.sort_values(["date"], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)
    df["past sales"] = df_grouped[col].apply(
        lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean()
    )
    return df


def ewma_merge(df_test, df_train, ewma_col, group_cols):
    def get_latest_ewmas(df):
        return df.loc[df["date"] == df["date"].max(), ewma_col]

    df_train_latest_ewma = (
        df_train[["date", ewma_col] + group_cols]
        .groupby(group_cols)
        .apply(get_latest_ewmas)
        .reset_index()
    )

    df_test = df_test.merge(
        df_train_latest_ewma[[ewma_col] + group_cols], on=group_cols, how="left"
    )

    return df_test


def seasonality_features(df):
    df["date"] = pd.to_datetime(df["date"])
    # df["weekday"] = df['date'].dt.dayofweek
    df["weekday"] = df["date"].dt.day_name()
    df["day in month"] = df["date"].dt.day
    df["day in year"] = df["date"].dt.dayofyear
    return df


def get_events(df):
    for event_date in ["2015-08-07", "2016-08-12", "2017-08-11"]:
        for event_days in range(0, 6):
            df.loc[
                df["date"]
                == str(
                    (pd.to_datetime(event_date) + datetime.timedelta(days=event_days))
                ).split(" ")[0],
                "days around Primer Grito de Independencia",
            ] = event_days
    return df


def predict(model, dataloader, df, min_gpt=True):
    model.eval()

    yhat = []
    for input_embeds, _ in dataloader:
        with torch.no_grad():
            if min_gpt:
                yhat += model.generate(input_embeds.to(device)).cpu().numpy().tolist()
            else:
                yhat += (
                    model.model(
                        inputs_embeds=input_embeds.to(torch.bfloat16).to(device)
                    )["logits"]
                    .cpu()
                    .float()
                    .numpy()
                    .tolist()
                )

    df["yhat"] = yhat if min_gpt else list(itertools.chain.from_iterable(yhat))
    df["yhat"] = np.clip(df["yhat"], 0, None)
    df["yhat"] = np.exp(df["yhat"]) - 1
    return df


def main(test, pretrained):
    np.random.seed(666)
    torch.manual_seed(42)

    folder_path = "data"
    train_dataset = TabularDataset(folder_path)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=64,
        num_workers=0,
    )

    hf_gpt_path = "openai-community/gpt2"
    output_model_dir = "saved_model"

    model = TabGPT(
        model_path=output_model_dir,
        tok_path=hf_gpt_path,
        lr=3e-4,
        pretrained=pretrained,
    ).to(device)

    # Initialize a trainer
    trainer = L.Trainer(
        max_epochs=1,
        precision="bf16-true",
        gradient_clip_val=1.0,
    )

    # trainer.fit(model, train_loader)

    # model.model.save_pretrained(output_model_dir)

    # inference on store-sales

    folder_path = "./finetuning/data"
    finetuning_dataset = TabularDataset(folder_path)

    finetuning_data_loader = DataLoader(
        finetuning_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=64,
        num_workers=0,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,  # Dropout rate
    )

    peft_model = get_peft_model(model, peft_config)

    model.model = peft_model
    model.model.train()

    trainer = L.Trainer(
        max_epochs=1,
        precision="bf16-true",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, finetuning_data_loader)

    # df_train, df_test, features, categorical_features, numerical_features = get_data()

    # features_embeds_test = get_column_embeddings(df_test, "daily store sales", categorical_features, numerical_features, number_of_cols=len(features))

    # test_dataset = TensorDataset(
    #     features_embeds_test,
    #     torch.tensor(df_test["sales_transformed"].tolist(), dtype=torch.float32)
    # )

    # df_test = predict(model, DataLoader(test_dataset, batch_size=32, shuffle=False), df_test, min_gpt=False)

    # evaluation(df_test["sales"], df_test["yhat"])
    # plot_timeseries(df_test, "val", True)
    # for pg in df_test["product group"].unique():
    #     if pg != "BREAD/BAKERY":
    #         plot_timeseries(df_test[df_test["product group"] == pg], pg + "_val", True)
    #     else:
    #         plot_timeseries(df_test[df_test["product group"] == pg], "BREAD_BAKERY_val", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    main(args.test, args.pretrained)
