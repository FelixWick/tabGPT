import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from tabgpt.callbacks import whole_epoch_train_loss
from tabgpt.data.store_sales.data_setup import StoreSalesData
from tabgpt.model_hf import tabGPT_HF, tabGPTConfig
from tabgpt.tabular_dataset import load_datasets
from tabgpt.utils import evaluation, predict
import torch
from torch.utils.data import TensorDataset, DataLoader

from tabgpt.model import tabGPT
from tabgpt.trainer import Trainer
from tabgpt.col_embed import Embedder
import os

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


def main():
    np.random.seed(666)
    torch.manual_seed(42)

    storesales = StoreSalesData()
    storesales.setup()
    n_cols = storesales.n_features

    embedder = Embedder(storesales)

    load_in_memory = True

    loaded = False
    if not loaded:
    
        embedder.train()
        features_embeds_train = embedder.embed(n_cols,save=False if load_in_memory else True, remove_first=True)

        embedder.val()
        features_embeds_val = embedder.embed(n_cols, save=False if load_in_memory else True, remove_first=True)

    df_train = storesales.df_train
    df_val = storesales.df_val

    target_col = storesales.main_target
    target_scaler = storesales.scaler[target_col]


    if load_in_memory:
        train_dataset = TensorDataset(
            features_embeds_train, 
            torch.tensor(df_train[target_col].tolist(), dtype=torch.float32)
            )
        
        val_dataset = TensorDataset(
            features_embeds_val, 
            torch.tensor(df_val[target_col].tolist(), dtype=torch.float32)
            )
    
    else:
        train_dataset =  load_datasets(dataset_loader=[storesales], mode='train',target=target_col)
        val_dataset =  load_datasets(dataset_loader=[storesales], mode='val', target=target_col)

 
    config = tabGPTConfig(n_layer=4, n_head=4, block_size=n_cols+1, n_output_nodes=1)
    model = tabGPT_HF(config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 1000000
    train_config.epochs = 1 # used in single training of concept paper
    # train_config.epochs = 89 # used in individual comparison for cross-training of concept paper
    train_config.num_workers = 0
    train_config.learning_rate = 1e-3
    train_config.batch_size = 64
    trainer = Trainer(train_config, model, train_dataset, target_scaler=target_scaler)
    trainer.set_callback('on_epoch_end', whole_epoch_train_loss)


    trainer.run()

    path = './store_sales_model'
    print("Storing trained model")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)

    df_val[target_col] = target_scaler.inverse_transform(df_val[[target_col]])
    df_val = predict(model, DataLoader(val_dataset, batch_size=32), df_val, target_scaler)
  
    evaluation(df_val["sales"], df_val["yhat"])
    plot_timeseries(df_val, "val", True)
    for pg in df_val["product group"].unique():
        if pg != "BREAD/BAKERY":
            plot_timeseries(df_val[df_val["product group"] == pg], pg + "_val", True)
        else:
            plot_timeseries(df_val[df_val["product group"] == pg], "BREAD_BAKERY_val", True)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
