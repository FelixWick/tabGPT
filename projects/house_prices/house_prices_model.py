import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split

from tabgpt.callbacks import whole_epoch_train_loss
from tabgpt.model_hf import tabGPT_HF, tabGPTConfig
import torch
from torch.utils.data import TensorDataset, DataLoader
from tabgpt.data.house_prices.data_setup import HousePricesData

import os
from tabgpt.model import tabGPT
from tabgpt.col_embed import Embedder
from tabgpt.trainer import Trainer

from tabgpt.utils import predict, evaluation

from IPython import embed


if torch.cuda.is_available():       
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def main():
    np.random.seed(666)
    torch.manual_seed(42)

    # use data from Kaggle competition https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
    house_prices = HousePricesData()
    house_prices.setup()
    df_train = house_prices.df_train

    target_column = house_prices.main_target
    target_scaler = house_prices.scaler[target_column]

    embedder = Embedder(house_prices)
    embedder.train()
    features_embeds_train = embedder.embed(n_cols=house_prices.n_features)

    train_dataset = TensorDataset(
        features_embeds_train,
        torch.tensor(house_prices.df_train[target_column].tolist(), dtype=torch.float32)
        )

    max_length = house_prices.n_features + 1

    n_layer, n_head = 4, 4 # gpt-micro
    config = tabGPTConfig(n_layer=n_layer, n_head=n_head, block_size=max_length, n_output_nodes=1)
    model = tabGPT_HF(config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 1000000
    train_config.epochs = 1 # used in single training of concept paper
    # train_config.epochs = 88 # used in individual comparison for cross-training of concept paper
    train_config.num_workers = 0
    train_config.batch_size = 64
    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_epoch_end', whole_epoch_train_loss)

    trainer.run()

    path = './house_prices_model'
    print("Storing trained model")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)

    # inference
    df_train = predict(model, DataLoader(train_dataset, batch_size=32), df_train, target_scaler=target_scaler)
    evaluation(df_train[target_column], df_train["yhat"])

    
    embedder.val()
    features_embeds_val = embedder.embed(n_cols=house_prices.n_features)
    test_dataset = TensorDataset(
        features_embeds_val,
        torch.tensor(house_prices.df_val[target_column].tolist(), dtype=torch.float32)
    )

    df_test = house_prices.df_val

    df_test = predict(model, DataLoader(test_dataset, batch_size=32), df_test, target_scaler=target_scaler)
    evaluation(df_test[target_column], df_test["yhat"])

    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()