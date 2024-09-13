import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime

from tabgpt.data.house_prices.data_setup import HousePricesData
from tabgpt.data.ny_bicycles.data_setup import NYBicyclesData
from tabgpt.data.simulated_demand.data_setup import SimulatedDemandData
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
        ts = df.groupby(['date'])[['target', 'yhat']].sum().reset_index()
    else:
        ts = df.groupby(['date'])['target'].sum().reset_index()
    plt.figure()
    ts.index = ts['date']
    ts['target'].plot(style='r', label="target")
    if include_preds:
        ts['yhat'].plot(style='b-.', label="predictions")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig("ts_{}.png".format(suffix))
    plt.clf()



def main(pretrained):
    np.random.seed(666)
    torch.manual_seed(42)

    store_sales = StoreSalesData()
    store_sales.setup()

    house_prices = HousePricesData()
    house_prices.setup()

    simulated_demand = SimulatedDemandData()
    simulated_demand.setup(low=True)

    ny_bicycles = NYBicyclesData()
    ny_bicycles.setup()

    datasets = [store_sales, house_prices, ny_bicycles, simulated_demand]

    msg = [f'train samples {d.name}: {len(d.df_train)}' for d in datasets]
    for msg in msg:
        print(msg)

    max_features = max([data.n_features for data in datasets])

    features_embeds_train = [Embedder(data, mode='train').embed(n_cols=max_features) for data in datasets]

    features_embeds_train = torch.cat(features_embeds_train, dim=0)

    targets_train = []
    for data in datasets:
        targets_train += data.df_train['target'].to_list()

    train_dataset = TensorDataset(
        features_embeds_train, 
        torch.tensor(targets_train, dtype=torch.float32)
        )

 
    model_config = tabGPT.get_default_config()
    model_config.model_type = 'gpt-micro'
    model_config.vocab_size = 50257 # openai's model vocabulary
    model_config.block_size = max_features + 1 # 1024 is openai's model block_size
    model_config.n_output_nodes = 1
    model = tabGPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 100000
    train_config.epochs = 80
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

    val_datasets = [TensorDataset(
        Embedder(d,mode='val').embed(n_cols=max_features), 
        torch.tensor(d.df_val["target"].tolist(), dtype=torch.float32)
        ) for d in datasets]

    df_val_store_sales = predict(model, DataLoader(val_datasets[0], batch_size=32), store_sales.df_val)
    print('Evaluating store sales...')
    evaluation(df_val_store_sales["target"], df_val_store_sales["yhat"])
    plot_timeseries(df_val_store_sales, "store_sales", True)

    df_val_house_prices = predict(model, DataLoader(val_datasets[1], batch_size=32), house_prices.df_val)
    print('Evaluating house prices...')
    evaluation(df_val_house_prices["target"], df_val_house_prices["yhat"])

    df_val_bicycles_count = predict(model, DataLoader(val_datasets[2], batch_size=32), ny_bicycles.df_val)
    print('Evaluating bicycles...')
    evaluation(df_val_bicycles_count["target"], df_val_bicycles_count["yhat"])
    plot_timeseries(df_val_bicycles_count, "bicycles_count", True)

    df_val_simulated_demand = predict(model, DataLoader(val_datasets[3], batch_size=32), simulated_demand.df_test)
    print('Evaluating simulated demand...')
    evaluation(df_val_simulated_demand["target"], df_val_simulated_demand["yhat"])
    plot_timeseries(df_val_simulated_demand, "simulated_demand", True)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    main(args.pretrained)
