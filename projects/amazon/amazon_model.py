import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tabgpt.callbacks import whole_epoch_train_loss
from tabgpt.data.amazon.data_setup import AmazonData
from tabgpt.utils import evaluation, predict
import torch
from torch.utils.data import TensorDataset, DataLoader

from tabgpt.model_hf import tabGPT_HF, tabGPTConfig
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
        ts = df.groupby(['date'])[['quantity', 'yhat']].sum().reset_index()
    else:
        ts = df.groupby(['date'])['quantity'].sum().reset_index()
    plt.figure()
    ts.index = ts['date']
    ts['quantity'].plot(style='r', label="quantity")
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

    amazon = AmazonData()
    amazon.setup()
    n_cols = amazon.n_features

    embedder = Embedder(amazon)

    embedder.train()
    features_embeds_train = embedder.embed(n_cols)

    embedder.val()
    features_embeds_val = embedder.embed(n_cols)

    # embedder.test()
    # features_embeds_test = embedder.embed(n_cols)

    df_train = amazon.df_train
    df_val = amazon.df_val
    # df_test = amazon.df_test


    train_dataset = TensorDataset(
        features_embeds_train,
        torch.tensor(df_train[amazon.target_column].tolist(), dtype=torch.float32)
    )

    max_length = n_cols +1

    # New tabGPT model
    n_layer, n_head = 4, 4 # gpt-micro
    config = tabGPTConfig(n_layer=n_layer, n_head=n_head, block_size=amazon.n_features + 1, n_output_nodes=1)
    model = tabGPT_HF(config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 1000000
    train_config.epochs = 100
    train_config.num_workers = 0
    train_config.batch_size = 64
    train_config.observe_train_loss = True
    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_epoch_end', whole_epoch_train_loss)

    trainer.run()

    # inference
    print(f"Train")
    df_train = predict(model, DataLoader(train_dataset, batch_size=32), df_train)
    evaluation(df_train[amazon.target_column], df_train["yhat"])
    plot_timeseries(df_train, "train", True)

    print(f"Valid")
    val_dataset = TensorDataset(
        features_embeds_val,
        torch.tensor(df_val[amazon.target_column].tolist(), dtype=torch.float32)
    )

    df_val = predict(model, DataLoader(val_dataset, batch_size=32), df_val)
    evaluation(df_val[amazon.target_column], df_val["yhat"])
    plot_timeseries(df_val, "val", True)

    # print(f"Test")
    # test_dataset = TensorDataset(
    #     features_embeds_test,
    #     torch.tensor(df_test[amazon.target_column].tolist(), dtype=torch.float32)
    # )

    # df_test = predict(model, DataLoader(test_dataset, batch_size=32), df_test)
    # evaluation(df_test[amazon.target_column], df_test["yhat"])
    # plot_timeseries(df_test, "test", True)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    main(args.test, False)
    # main(args.test, args.pretrained)
