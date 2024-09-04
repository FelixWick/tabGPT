import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
import matplotlib.pyplot as plt
import itertools
import lightning as L
import os

from preprocess import (
    get_data_bicycles_count,
    get_data_house_prices,
    get_data_simulated_demand,
    get_data_store_sales,
)

from tabgpt.pl_model import TabGPT
from tabgpt.tabular_dataset import TabularDataset, load_datasets
import torch
from torch.utils.data import TensorDataset, DataLoader

from tabgpt.model import tabGPT
from tabgpt.trainer import Trainer
from tabgpt.col_embed import get_column_embeddings

from IPython import embed

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def plot_timeseries(df, suffix, include_preds=False):
    if include_preds:
        ts = df.groupby(["date"])[["target", "yhat"]].sum().reset_index()
    else:
        ts = df.groupby(["date"])["target"].sum().reset_index()
    plt.figure()
    ts.index = ts["date"]
    ts["target"].plot(style="r", label="target")
    if include_preds:
        ts["yhat"].plot(style="b-.", label="predictions")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig("ts_{}.png".format(suffix))
    plt.clf()


def evaluation(y, yhat):
    print("RMSLE: ", root_mean_squared_log_error(y, yhat))
    print("mean(y): ", np.mean(y))


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


def main(finetune_on):
    np.random.seed(666)
    torch.manual_seed(42)

    datasets = ["store-sales", "house-prices", "demand-forecasting", "ny-bicycles"]

    # Parse the topics
    dataset_names = [dataset for dataset in datasets if dataset != args.finetune_on]

    # Define the root directory of your dataset
    root_dir = "data"

    # Load the specified datasets
    datasets = load_datasets(root_dir, dataset_names)

    # Combine the datasets
    combined_dataset = datasets[0]
    for dataset in datasets[1:]:
        combined_dataset += dataset

    # create Dataloader
    train_loader = DataLoader(
        combined_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=64,
        num_workers=0,
    )

    path = f"./saved_model_for_finetuning_on_{args.finetune_on}"
    local_model_path = None if not os.path.isdir(path) else path

    if local_model_path is not None:
        print("Loading pretrained model for finetuning")
    else:
        print("Starting pretraining")

    # create Lightning model
    # from scratch if path is None, else PEFT Lora model
    model = TabGPT(
        lr=2e-5,
        local_model_path=local_model_path,
    ).to(device)

    # Pretrain only if not done already
    if local_model_path is None:
        # Initialize a trainer
        trainer = L.Trainer(
            max_epochs=12,
            precision="bf16-true",
            gradient_clip_val=1.0,
            # accumulate_grad_batches=8,
            # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)]
        )

        trainer.fit(model, train_loader)

        print("Storing trained model")
        os.makedirs(path, exist_ok=True)
        model.model.save_pretrained(path)

    else:  # finetune
        torch_dataset = TabularDataset(f"data/{args.finetune_on}")
        # create Dataloader
        finetuning_train_loader = DataLoader(
            torch_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=64,
            num_workers=0,
        )

        # Initialize a trainer
        trainer = L.Trainer(
            max_epochs=50,
            precision="bf16-true",
            gradient_clip_val=1.0,
        )

        trainer.fit(model, finetuning_train_loader)
        model = model.to(device)

        max_features = 10

        if args.finetune_on == "store-sales":
            (
                df_train,
                df_val_store_sales,
                features,
                categorical_features,
                numerical_features,
            ) = get_data_store_sales()

            features_embeds_val_store_sales = get_column_embeddings(
                df_val_store_sales,
                "store sales",
                categorical_features,
                numerical_features,
                number_of_cols=max_features,
            )

            val_dataset_store_sales = TensorDataset(
                features_embeds_val_store_sales,
                torch.tensor(
                    df_val_store_sales["target"].tolist(), dtype=torch.float32
                ),
            )

            df_val_store_sales = predict(
                model,
                DataLoader(val_dataset_store_sales, batch_size=32),
                df_val_store_sales,
                min_gpt=False,
            )
            evaluation(df_val_store_sales["target"], df_val_store_sales["yhat"])
            plot_timeseries(df_val_store_sales, "store_sales", True)

        if args.finetune_on == "house-prices":
            (
                df_train,
                df_val_house_prices,
                features,
                categorical_features,
                numerical_features,
            ) = get_data_house_prices()

            features_embeds_val_house_prices = get_column_embeddings(
                df_val_house_prices,
                "house prices",
                categorical_features,
                numerical_features,
                number_of_cols=max_features,
            )

            val_dataset_house_prices = TensorDataset(
                features_embeds_val_house_prices,
                torch.tensor(
                    df_val_house_prices["target"].tolist(), dtype=torch.float32
                ),
            )

            df_val_house_prices = predict(
                model,
                DataLoader(val_dataset_house_prices, batch_size=32),
                df_val_house_prices,
                min_gpt=False,
            )
            evaluation(df_val_house_prices["target"], df_val_house_prices["yhat"])

        if args.finetune_on == "demand-forecasting":
            (
                df_train,
                df_val_simulated_demand,
                features,
                categorical_features,
                numerical_features,
            ) = get_data_simulated_demand()

            features_embeds_val_simulated_demand = get_column_embeddings(
                df_val_simulated_demand,
                "retail demand forecasting",
                categorical_features,
                numerical_features,
                number_of_cols=max_features,
            )
            val_dataset_simulated_demand = TensorDataset(
                features_embeds_val_simulated_demand,
                torch.tensor(
                    df_val_simulated_demand["target"].tolist(), dtype=torch.float32
                ),
            )

            df_val_simulated_demand = predict(
                model,
                DataLoader(val_dataset_simulated_demand, batch_size=32),
                df_val_simulated_demand,
                min_gpt=False,
            )
            evaluation(df_val_simulated_demand["target"], df_val_simulated_demand["yhat"])
            plot_timeseries(df_val_simulated_demand, "simulated_demand", True)

        if args.finetune_on == "ny-bicycles":
            (
                df_train,
                df_val_bicycles_count,
                features,
                categorical_features,
                numerical_features,
            ) = get_data_bicycles_count()

            features_embeds_val_bicycles_count = get_column_embeddings(
                df_val_bicycles_count,
                "bicycles count",
                categorical_features,
                numerical_features,
                number_of_cols=max_features,
            )

            val_dataset_bicycles_count = TensorDataset(
                features_embeds_val_bicycles_count,
                torch.tensor(
                    df_val_bicycles_count["target"].tolist(), dtype=torch.float32
                ),
            )

            df_val_bicycles_count = predict(
                model,
                DataLoader(val_dataset_bicycles_count, batch_size=32),
                df_val_bicycles_count,
                min_gpt=False,
            )
            evaluation(df_val_bicycles_count["target"], df_val_bicycles_count["yhat"])
            plot_timeseries(df_val_bicycles_count, "bicycles_count", True)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetune_on", type=str, required=True, help="dataset-name, e.g. house-prices"
    )
    args = parser.parse_args()
    main(args.finetune_on)
