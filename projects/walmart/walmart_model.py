import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
import matplotlib.pyplot as plt
import itertools
import lightning as L
import os

from tabgpt.callbacks import whole_epoch_train_loss
from tabgpt.data.walmart.data_setup import WalmartData
from tabgpt.model_hf import tabGPT_HF, tabGPTConfig
from tabgpt.tabular_dataset import TabularDataset, load_datasets
from tabgpt.utils import evaluation, predict
import torch
from torch.utils.data import TensorDataset, DataLoader


from tabgpt.trainer import Trainer
from tabgpt.col_embed import Embedder

from IPython import embed

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


if __name__ == '__main__':

    pretrained_model_path = './pretrained_walmart_model'

    wm = WalmartData()
    wm.setup(last_nrows=1000) # only test
    emb = Embedder(wm)

    in_memory = False
    files_generated = False # only used in file-approach

    if in_memory:
        print('Generate embedding and load into memory')
        # in-memory approach (re-run every time)
        emb.train()
        f_embeds_train = emb.embed(n_cols=wm.n_features)
        emb.val()
        f_embeds_val = emb.embed(n_cols=wm.n_features)

    else:
        # file-approach (only need to run once)
        if not files_generated:
            print('Storing data as files')
            emb.train()
            emb.embed(n_cols=wm.n_features,save=True)
            emb.val()
            emb.embed(n_cols=wm.n_features,save=True)
    

    print(f'train samples: {len(wm.df_train)}')

    n_layer, n_head = 4, 4 # gpt-micro
    config = tabGPTConfig(n_layer=n_layer, n_head=n_head, block_size=wm.n_features + 1, n_output_nodes=1)
    model = tabGPT_HF(config)

    target_column = wm.main_target
    train_targets = wm.df_train[wm.main_target]
    val_targets = wm.df_val[wm.main_target]
    target_scaler = wm.scaler[target_column]

    df_train = wm.df_train
    df_val = wm.df_val


    if in_memory:
        train_dataset = TensorDataset(
            f_embeds_train,
            torch.tensor(train_targets.tolist(), dtype=torch.float32)
            )
    else:
        train_dataset =  load_datasets(dataset_loader=[wm], mode='train', target=target_column)


    train_config = Trainer.get_default_config()
    train_config.epochs = 1
    train_config.num_workers = 0
    train_config.batch_size = 64
    train_config.learning_rate = 5e-4
    trainer = Trainer(train_config, model, train_dataset)

    trainer.run()

    print("Storing trained model")
    os.makedirs(pretrained_model_path, exist_ok=True)
    model.save_pretrained(pretrained_model_path)

    # inference
    df_train[target_column] = target_scaler.inverse_transform(df_train[[target_column]])

    df_train = predict(model, DataLoader(train_dataset, batch_size=32), wm.df_train, target_scaler=target_scaler)
    evaluation(df_train[target_column], df_train["yhat"])

    if in_memory:
        val_dataset = TensorDataset(
            f_embeds_val,
            torch.tensor(val_targets.tolist(), dtype=torch.float32)
        )
    else:
        val_dataset =  load_datasets(dataset_loader=[wm], mode='val', target=target_column)


    df_val[target_column] = target_scaler.inverse_transform(df_val[[target_column]])

    df_val = predict(model, DataLoader(val_dataset, batch_size=32), wm.df_val, target_scaler=target_scaler)
    evaluation(df_val[target_column], df_val["yhat"])


