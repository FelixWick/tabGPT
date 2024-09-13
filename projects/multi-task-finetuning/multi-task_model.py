import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
import matplotlib.pyplot as plt
import itertools
import lightning as L
import os


from peft import LoraConfig, TaskType, get_peft_model


from tabgpt.callbacks import whole_epoch_train_loss
from tabgpt.data.house_prices.data_setup import HousePricesData
from tabgpt.data.ny_bicycles.data_setup import NYBicyclesData
from tabgpt.data.simulated_demand.data_setup import SimulatedDemandData
from tabgpt.data.store_sales.data_setup import StoreSalesData
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

def get_all_linear(model):
    # Create a list to store the layer names
    layer_names = []
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear)):
            # model name parsing 
            layer_names.append(name)
            #layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    
    return layer_names


def make_lora_model(model):

    #all_linear = get_all_linear(model)
    # lm_head = all_linear[-1]
    peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules='all_linear',
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,  # Dropout rate
        )

    return get_peft_model(model, peft_config)


def construct_files(data_list):

    for d in data_list:
        if d.name == 'house_prices':
            d.setup(all_features=False)
        else:
            d.setup()
        if d.name == 'simulated_demand':
            d.df_val = d.df_test
    
    max_features = max([d.n_features for d in data_list])

    for d in data_list:
        Embedder(d, mode='train').embed(n_cols=max_features, save=True)
        Embedder(d, mode='val').embed(n_cols=max_features, save=True)


def main(finetune_on):
    np.random.seed(666)
    torch.manual_seed(42)

    data_loader = [StoreSalesData(), HousePricesData(), NYBicyclesData(), SimulatedDemandData()]
    
    files_generated = False

    if not files_generated:
        construct_files(data_list=data_loader)

    else:
        for d in data_loader:
            if d.name == 'house_prices':
                d.setup(all_features=False)
            else:
                d.setup()
            if d.name == 'simulated_demand':
                d.df_val = d.df_test

    target_column = 'target'

    msg = [f'train samples {d.name}: {len(d.df_train)}' for d in data_loader]
    for msg in msg:
        print(msg)

    max_features = max([data.n_features for data in data_loader])
    print('Max features:', max_features)

    max_pretrain_lr = 5e-4
    max_finetune_lr = 1e-4

    n_layer, n_head = 4, 4 # gpt-micro
    config = tabGPTConfig(n_layer=n_layer, n_head=n_head, block_size=max_features+1, n_output_nodes=1)

    # Parse the topics
    data_loader_to_pretrain = [d for d in data_loader if d.name != args.finetune_on]

    # Load the specified datasets
    datasets = load_datasets(dataset_loader=data_loader_to_pretrain, mode='train')

    # Combine the datasets
    combined_dataset = datasets[0]
    for dataset in datasets[1:]:
        combined_dataset += dataset

    path = f"./saved_model_for_finetuning_on_{args.finetune_on}"
    local_model_path = None if not os.path.isdir(path) else path

    if local_model_path is not None:
        print("Loading pretrained model for finetuning")
    else:
        print("Starting pretraining")

    # create Lightning model
    # from scratch if path is None, else PEFT Lora model
    config = tabGPTConfig(n_layer=4, n_head=4, block_size=max_features+1, n_output_nodes=1)
    model = tabGPT_HF(config)

    # Pretrain only if not done already
    if local_model_path is None:
        train_config = Trainer.get_default_config()
        train_config.epochs = 30
        train_config.num_workers = 0
        train_config.batch_size = 64
        train_config.learning_rate = max_pretrain_lr
        trainer = Trainer(train_config, model, combined_dataset)

        trainer.run()

        print("Storing trained model")
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)

    else:  # finetune
        config = tabGPTConfig(n_layer=4, n_head=4, block_size=max_features+1, n_output_nodes=1)
        model = tabGPT_HF(config)
        model = model.from_pretrained(path)
        lora_model = make_lora_model(model)
    
        data_loader_to_finetune = [d for d in data_loader if d.name == args.finetune_on]

        torch_finetune_dataset =  load_datasets(dataset_loader=data_loader_to_finetune, mode='train')[0]

        finetune_config = Trainer.get_default_config()
        finetune_config.epochs = 1
        finetune_config.num_workers = 0
        finetune_config.batch_size = 64
        finetune_config.learning_rate = max_finetune_lr
        trainer = Trainer(finetune_config, lora_model, torch_finetune_dataset, use_scheduler=True)
        trainer.set_callback('on_epoch_end', whole_epoch_train_loss)
        trainer.run()


        val_datasets = {d.name: TensorDataset(
                Embedder(d, mode='val').embed(n_cols=max_features),
                torch.tensor(
                    d.df_val[d.target_column].tolist(), dtype=torch.float32
                ),
            ) for d in data_loader}
        
        d_dict = {d.name: d for d in data_loader}

        if args.finetune_on == "store_sales":

            df_val_store_sales = predict(
                lora_model,
                DataLoader(val_datasets['store_sales'], batch_size=32),
                d_dict['store_sales'].df_val,
            )
            evaluation(df_val_store_sales[target_column], df_val_store_sales["yhat"])
            plot_timeseries(df_val_store_sales, "store_sales", True)

        if args.finetune_on == "house_prices":

            df_val_house_prices = predict(
                lora_model,
                DataLoader(val_datasets['house_prices'], batch_size=32),
                d_dict['house_prices'].df_val,
            )
            evaluation(df_val_house_prices[target_column], df_val_house_prices["yhat"])

        if args.finetune_on == "simulated_demand":
            
            df_val_simulated_demand = predict(
                lora_model,
                DataLoader(val_datasets['simulated_demand'], batch_size=32),
                d_dict['simulated_demand'].df_val,
            )
            evaluation(df_val_simulated_demand[target_column], df_val_simulated_demand["yhat"])
            plot_timeseries(df_val_simulated_demand, "simulated_demand", True)

        if args.finetune_on == "ny_bicycles":

            df_val_bicycles_count = predict(
                lora_model,
                DataLoader(val_datasets["ny_bicycles"], batch_size=32),
                d_dict['ny_bicycles'].df_val,
            )
            evaluation(df_val_bicycles_count[target_column], df_val_bicycles_count["yhat"])
            plot_timeseries(df_val_bicycles_count, "bicycles_count", True)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetune_on", type=str, required=True, help="dataset-name, e.g. house-prices"
    )
    args = parser.parse_args()
    main(args.finetune_on)
