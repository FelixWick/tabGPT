import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

from tabgpt.model import tabGPT
from tabgpt.trainer import Trainer
from tabgpt.col_embed import get_column_embeddings
from projects.house_prices.construct_dataset import construct_text

from IPython import embed


if torch.cuda.is_available():       
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def evaluation(y, yhat):
    print('RMSLE: ', root_mean_squared_log_error(y, yhat))
    print('mean(y): ', np.mean(y))


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


def main(test, enrich):
    np.random.seed(666)
    torch.manual_seed(42)

    # use data from Kaggle competition https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
    df_train_full = pd.read_csv("train.csv")

    # used in individual comparison for cross-training of concept paper
    # important_cols = ["OverallQual",
    #                   "GarageCars",
    #                   "ExterQual",
    #                   "Neighborhood",
    #                   "GrLivArea",
    #                   "GarageArea",
    #                   "BsmtQual",
    #                   "YearBuilt",
    #                   "KitchenQual",
    #                   "TotalBsmtSF",
    #                   ]
    # important_cols.append('Id')
    # important_cols.append('SalePrice')
    # df_train_full = df_train_full[important_cols]

    if enrich:
        df_train_full = construct_text(df_train_full)

    categorical_features = []
    numerical_features = []
    for col in df_train_full.drop(columns=["Id", "SalePrice"]).columns:
        if df_train_full[col].dtype == 'O':
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    features = categorical_features + numerical_features

    df_train_full["SalePrice_transformed"] = np.log(1 + df_train_full["SalePrice"])

    df_train_full = df_train_full[['Id', 'SalePrice', 'SalePrice_transformed'] + features]

    if test:
        df_test = pd.read_csv("test.csv")
        # df_test = df_test[important_cols[:-1]]
        if enrich:
            df_test = construct_text(df_test)       
        df_test = df_test[['Id'] + features]
        df_train = df_train_full
    else:
        df_train, df_test = train_test_split(df_train_full, test_size=0.2, random_state=666)

    num_max = df_train[numerical_features].abs().max()
    df_train[numerical_features] = df_train[numerical_features] / num_max
    df_test[numerical_features] = df_test[numerical_features] / num_max

    features_embeds_train = get_column_embeddings(df_train, "house prices", categorical_features, numerical_features, number_of_cols=len(features))

    train_dataset = TensorDataset(
        features_embeds_train,
        torch.tensor(df_train["SalePrice_transformed"].tolist(), dtype=torch.float32)
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
    train_config.epochs = 221
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
    evaluation(df_train["SalePrice"], df_train["yhat"])

    features_embeds_val = get_column_embeddings(df_test, "house prices", categorical_features, numerical_features, number_of_cols=len(features))

    if test:
        test_dataset = TensorDataset(
            features_embeds_val,
            torch.tensor(df_test["Id"].tolist(), dtype=torch.float32)
        )
    else:
        test_dataset = TensorDataset(
            features_embeds_val,
            torch.tensor(df_test["SalePrice_transformed"].tolist(), dtype=torch.float32)
        )

    df_test = predict(model, DataLoader(test_dataset, batch_size=32), df_test)
    if test:
        df_test[["Id", "yhat"]].rename(columns={"yhat": "SalePrice"}).to_csv("submission.csv", index=False)
    else:
        evaluation(df_test["SalePrice"], df_test["yhat"])

    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--enrich", action="store_true")
    args = parser.parse_args()
    main(args.test, args.enrich)
