import sys

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
import matplotlib.pyplot as plt

from tabgpt.callbacks import whole_epoch_train_loss
from tabgpt.data.ny_bicycles.data_setup import NYBicyclesData
import torch
from torch.utils.data import TensorDataset, DataLoader

from tabgpt.model import tabGPT
from tabgpt.trainer import Trainer
from tabgpt.col_embed import Embedder
from tabgpt.utils import evaluation, predict

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


def main(args):
    np.random.seed(666)
    torch.manual_seed(42)

    # use data from https://data.cityofnewyork.us/Transportation/Bicycle-Counts-for-East-River-Bridges-Historical-/gua4-p9wg/about_data
    nybicycles = NYBicyclesData()
    nybicycles.setup()
    n_cols = nybicycles.n_features

    embedder = Embedder(nybicycles)

    embedder.train()
    features_embeds_train = embedder.embed(n_cols)

    embedder.val()
    features_embeds_val = embedder.embed(n_cols)

    df_train = nybicycles.df_train
    df_val = nybicycles.df_val

    target_col = nybicycles.target_column


    train_dataset = TensorDataset(
        features_embeds_train, 
        torch.tensor(df_train[target_col].tolist(), dtype=torch.float32)
        )

    val_dataset = TensorDataset(
        features_embeds_val, 
        torch.tensor(df_val[target_col].tolist(), dtype=torch.float32)
        )

    # tabGPT model
    model_config = tabGPT.get_default_config()
    model_config.model_type = 'gpt-micro'
    model_config.vocab_size = 50257 # openai's model vocabulary
    model_config.block_size = n_cols+1 # 1024 is openai's model block_size
    model_config.n_output_nodes = 1
    model = tabGPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 100000
    train_config.epochs = 82 # used in individual comparison for cross-training of concept paper
    train_config.num_workers = 0
    train_config.batch_size = 64
    train_config.observe_train_loss = True
    trainer = Trainer(train_config, model, train_dataset ,log_dir='./logs')
    trainer.set_callback('on_epoch_end', whole_epoch_train_loss)


    # if train_config.observe_train_loss:
    #     def epoch_end_callback(trainer):
    #         print(f"epoch {trainer.epoch}: train loss {np.sqrt(trainer.aggregated_loss.detach().cpu())}")
    #     trainer.set_callback('on_epoch_end', epoch_end_callback)
    # else:
    #     def batch_end_callback(trainer):
    #         if trainer.iter_num % 100 == 0:
    #             print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    #     trainer.set_callback('on_batch_end', batch_end_callback)

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