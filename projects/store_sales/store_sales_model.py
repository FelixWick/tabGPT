import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from tabgpt.data.store_sales.data_setup import StoreSalesData
from tabgpt.utils import evaluation, predict
import torch
from torch.utils.data import TensorDataset, DataLoader

from tabgpt.model import tabGPT
from tabgpt.trainer import Trainer
from tabgpt.col_embed import Embedder

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


def main(test, pretrained):
    np.random.seed(666)
    torch.manual_seed(42)

    storesales = StoreSalesData()
    storesales.setup()
    n_cols = storesales.n_features

    embedder = Embedder(storesales)

    embedder.train()
    features_embeds_train = embedder.embed(n_cols)

    embedder.val()
    features_embeds_val = embedder.embed(n_cols)

    df_train = storesales.df_train
    df_val = storesales.df_val


    train_dataset = TensorDataset(
        features_embeds_train,
        torch.tensor(df_train[storesales.target_column].tolist(), dtype=torch.float32)
    )

    max_length = n_cols +1

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
    train_config.epochs = 94 # used in single training of concept paper
    # train_config.epochs = 89 # used in individual comparison for cross-training of concept paper
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

    if test:
        storesales.test_setup()
        df_test = storesales.df_test
        embedder.test()
        features_embeds_test = embedder.embed(storesales.n_features)
        test_dataset = TensorDataset(
            features_embeds_test,
            torch.tensor(storesales.df_test[storesales.target_column].tolist(), dtype=torch.float32)
        )
    else:
        test_dataset = TensorDataset(
            features_embeds_val,
            torch.tensor(storesales.df_val[storesales.target_column].tolist(), dtype=torch.float32)
        )

    df_test = df_test if test else df_val
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
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    main(args.test, args.pretrained)
