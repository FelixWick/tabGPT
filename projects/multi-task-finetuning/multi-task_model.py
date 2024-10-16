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
from tabgpt.data.babies_r_us.data_setup import BabiesData
from tabgpt.data.buybuy_baby.data_setup import BuyBuyBabyData
from tabgpt.data.favorita.data_setup import FavoritaData
from tabgpt.data.house_prices.data_setup import HousePricesData
from tabgpt.data.ny_bicycles.data_setup import NYBicyclesData
from tabgpt.data.rossmann.data_setup import RossmannData
from tabgpt.data.simulated_demand.data_setup import SimulatedDemandData
from tabgpt.data.store_sales.data_setup import StoreSalesData
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
    device = torch.device("cuda:0") # RTX 4090
    print("Using GPU.")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def plot_timeseries(df, suffix, target_col, include_preds=False):
    if include_preds:
        ts = df.groupby(["date"])[[target_col, "yhat"]].sum().reset_index()
    else:
        ts = df.groupby(["date"])[target_col].sum().reset_index()
    plt.figure()
    ts.index = ts["date"]
    ts[target_col].plot(style="r", label="target")
    if include_preds:
        ts["yhat"].plot(style="b-.", label="predictions")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig("./plots/ts_{}.png".format(suffix))
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

def n_train_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_lora_model(model):

    all_linear = get_all_linear(model)
    # lm_head = all_linear[-1]
    peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=all_linear[:-1],
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
        elif d.name == 'simulated_demand':
            d.setup()
            d.df_val = d.df_test
        elif d.name == 'store_sales':
            d.setup()
        else:
            d.setup(last_nrows=200_000)
            
    
    max_features = max([d.n_features for d in data_list])

    for d in data_list:
        Embedder(d, mode='train').embed(n_cols=max_features, save=True)
        Embedder(d, mode='val').embed(n_cols=max_features, save=True)


def main(finetune_on):
    np.random.seed(666)
    torch.manual_seed(42)

    data_loader = [StoreSalesData(), 
                   HousePricesData(), 
                   NYBicyclesData(), 
                   SimulatedDemandData(), 
                   FavoritaData(), 
                   RossmannData(), 
                   WalmartData(), 
                   BabiesData(),
                   BuyBuyBabyData(),
                   ]
    
    files_generated = False

    if not files_generated:
        construct_files(data_list=data_loader)

    else:
        for d in data_loader:
            if d.name == 'house_prices':
                d.setup(all_features=False)
            elif d.name == 'simulated_demand':
                d.setup()
                d.df_val = d.df_test
            else: 
                d.setup()

    for d in data_loader:
        print(d)
        print('\n')

    max_features = max([data.n_features for data in data_loader])
    print('Max features:', max_features)

    embed()

    max_pretrain_lr = 5e-4
    max_finetune_lr = 1e-4

    n_layer, n_head = 4, 4 # gpt-micro
    config = tabGPTConfig(n_layer=n_layer, n_head=n_head, block_size=max_features+1, n_output_nodes=1)
    model = tabGPT_HF(config)

    # Parse the topicsnvisi
    data_loader_to_pretrain = [d for d in data_loader if d.name != args.finetune_on]

    # Load the specified datasets
    combined_dataset = load_datasets(dataset_loader=data_loader_to_pretrain, mode='train', only_main=True)

    path = f"./saved_model_for_finetuning_on_{args.finetune_on}"
    local_model_path = None if not os.path.isdir(path) else path

    if local_model_path is not None:
        print("Loading pretrained model for finetuning")
    else:
        print("Starting pretraining")

    # Pretrain only if not done already
    if local_model_path is None:
        train_config = Trainer.get_default_config()
        train_config.epochs = 50
        train_config.num_workers = 10
        train_config.batch_size = 64
        train_config.learning_rate = max_pretrain_lr
        trainer = Trainer(train_config, model, combined_dataset, progress_bar=False)

        trainer.run()

        print("Storing trained model")
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)

    else:  # finetune
        model = tabGPT_HF(config)
        model = model.from_pretrained(path)
        lora_model = make_lora_model(model)

        print(f'{(n_train_params(lora_model)/n_train_params(model)) * 100}% of all params trained')
    
        data_loader_to_finetune = [d for d in data_loader if d.name == args.finetune_on]

        torch_finetune_dataset =  load_datasets(dataset_loader=data_loader_to_finetune, mode='train',target=data_loader_to_finetune[0].main_target)

        finetune_config = Trainer.get_default_config()
        finetune_config.epochs = 10
        finetune_config.num_workers = 10
        finetune_config.batch_size = 64
        finetune_config.learning_rate = max_finetune_lr
        trainer = Trainer(finetune_config, model, torch_finetune_dataset, use_scheduler=True)
        trainer.set_callback('on_epoch_end', whole_epoch_train_loss)
        trainer.run()


        val_datasets = {d.name: load_datasets(dataset_loader=[d], mode='val', target=d.main_target) for d in data_loader}
        
        d_dict = {d.name: d for d in data_loader_to_finetune}

        if args.finetune_on == "store_sales":

            target_col = d_dict['store_sales'].main_target
            target_scaler = d_dict['store_sales'].scaler[target_col]
            df_val = d_dict['store_sales'].df_val
            df_val[target_col] = target_scaler.inverse_transform(df_val[[target_col]])

            df_val_store_sales = predict(
                model,
                DataLoader(val_datasets['store_sales'], batch_size=32),
                df_val,
                target_scaler
            )
            evaluation(df_val_store_sales[target_col], df_val_store_sales["yhat"])
            plot_timeseries(df_val_store_sales, "store_sales", target_col, True)
            for pg in df_val["product group"].unique():
                if pg != "BREAD/BAKERY":
                    plot_timeseries(df_val[df_val["product group"] == pg], pg + "_val", target_col, True)
                else:
                    plot_timeseries(df_val[df_val["product group"] == pg], "BREAD_BAKERY_val", target_col, True)

        if args.finetune_on == "house_prices":

            target_col = d_dict['house_prices'].main_target
            target_scaler = d_dict['house_prices'].scaler[target_col]
            df_val = d_dict['house_prices'].df_val
            df_val[target_col] = target_scaler.inverse_transform(df_val[[target_col]])

            df_val_house_prices = predict(
                model,
                DataLoader(val_datasets['house_prices'], batch_size=32),
                df_val,
                target_scaler
            )
            evaluation(df_val_house_prices[target_col], df_val_house_prices["yhat"])

        if args.finetune_on == "simulated_demand":

            target_col = d_dict['simulated_demand'].main_target
            target_scaler = d_dict['simulated_demand'].scaler[target_col]
            df_val = d_dict['simulated_demand'].df_val
            df_val[target_col] = target_scaler.inverse_transform(df_val[[target_col]])
            
            df_val_simulated_demand = predict(
                model,
                DataLoader(val_datasets['simulated_demand'], batch_size=32),
                df_val,
                target_scaler
            )
            evaluation(df_val_simulated_demand[target_col], df_val_simulated_demand["yhat"])
            plot_timeseries(df_val_simulated_demand, "simulated_demand", target_col, True)

        if args.finetune_on == "ny_bicycles":

            target_col = d_dict['ny_bicycles'].main_target
            target_scaler = d_dict['ny_bicycles'].scaler[target_col]
            df_val = d_dict['ny_bicycles'].df_val
            df_val[target_col] = target_scaler.inverse_transform(df_val[[target_col]])

            df_val_bicycles_count = predict(
                model,
                DataLoader(val_datasets["ny_bicycles"], batch_size=32),
                df_val,
                target_scaler
            )
            evaluation(df_val_bicycles_count[target_col], df_val_bicycles_count["yhat"])
            plot_timeseries(df_val_bicycles_count, "bicycles_count", target_col, True)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetune_on", type=str, required=True, help="dataset-name, e.g. house-prices"
    )
    args = parser.parse_args()
    main(args.finetune_on)
