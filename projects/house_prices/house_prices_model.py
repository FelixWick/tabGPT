import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
from tabgpt.data.house_prices.data_setup import HousePricesData

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


def main(test, pretrained):
    np.random.seed(666)
    torch.manual_seed(42)

    # use data from Kaggle competition https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
    house_prices = HousePricesData()
    house_prices.setup()
    df_train = house_prices.df_train

    embedder = Embedder(house_prices)
    embedder.train()
    features_embeds_train = embedder.embed(n_cols=house_prices.n_features)

    train_dataset = TensorDataset(
        features_embeds_train,
        torch.tensor(house_prices.df_train[house_prices.target_column].tolist(), dtype=torch.float32)
        )

    max_length = house_prices.n_features + 1

    # tabGPT model
    if pretrained:
        model = tabGPT.from_pretrained('gpt2', 1)
    else:
        model_config = tabGPT.get_default_config()
        model_config.model_type = 'gpt-micro'
        model_config.vocab_size = 50257 # openai's model vocabulary
        model_config.block_size = max_length # 1024 is openai's model block_size
        model_config.n_output_nodes = 1
        model = tabGPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 1000000
    train_config.epochs = 160 # used in single training of concept paper
    # train_config.epochs = 88 # used in individual comparison for cross-training of concept paper
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


    if test:
        house_prices.test_setup(all_train_data=False)
        embedder.test()
        features_embeds_test = embedder.embed(n_cols=house_prices.n_features)
        test_dataset = TensorDataset(
            features_embeds_test,
            torch.tensor(house_prices.df_test[house_prices.target_column].tolist(), dtype=torch.float32)
        )
    else:
        embedder.val()
        features_embeds_val = embedder.embed(n_cols=house_prices.n_features)
        test_dataset = TensorDataset(
            features_embeds_val,
            torch.tensor(house_prices.df_val[house_prices.target_column].tolist(), dtype=torch.float32)
        )

    df_test = house_prices.df_test if test else house_prices.df_val

    df_test = predict(model, DataLoader(test_dataset, batch_size=32), df_test)
    if test:
        df_test[["Id", "yhat"]].rename(columns={"yhat": "SalePrice"}).to_csv("submission.csv", index=False)
    else:
        evaluation(df_test["SalePrice"], df_test["yhat"])

    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    main(args.test, args.pretrained)
