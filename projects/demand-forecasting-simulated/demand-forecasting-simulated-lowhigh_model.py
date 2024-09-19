import sys

import numpy as np
import matplotlib.pyplot as plt

from tabgpt.data.simulated_demand.data_setup import SimulatedDemandData
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
        ts = df.groupby(['DATE'])[['target', 'yhat']].sum().reset_index()
    else:
        ts = df.groupby(['DATE'])['target'].sum().reset_index()
    plt.figure()
    ts.index = ts['DATE']
    ts['target'].plot(style='r', label="sales")
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

    simulated_demand_low = SimulatedDemandData(task_description='low sales')
    simulated_demand_low.setup(low=True)
    simulated_demand_high = SimulatedDemandData(task_description='high sales')
    simulated_demand_high.setup(low=False)

    n_cols = max(simulated_demand_low.n_features, simulated_demand_high.n_features)

    target_column = simulated_demand_low.target_column

    datasets = [simulated_demand_low, simulated_demand_high]
    features_embeds_train_list = [Embedder(d, mode='train').embed(n_cols=n_cols) for d in datasets]

    if args and args[0] == "--mode":
        mode = args[1]
    else:
        mode = "train_together"

    if mode == "train_together":
        features_embeds_train = torch.cat(
            (
                features_embeds_train_list
            ),
            dim=0,
        )

        max_length = n_cols + 1

        targets_train = (
            simulated_demand_low.df_train[target_column].tolist() + simulated_demand_high.df_train[target_column].tolist()
        )

        train_dataset = TensorDataset(
            features_embeds_train, torch.tensor(targets_train, dtype=torch.float32)
        )

    elif mode == "train_low":
        features_embeds_train = features_embeds_train_list[0]

        max_length = simulated_demand_low.n_features + 1

        targets_train = simulated_demand_low.df_train[target_column].tolist()

    elif mode == "train_high":
        features_embeds_train = features_embeds_train_list[1]

        max_length = simulated_demand_high.n_features + 1

        targets_train = simulated_demand_high.df_train[target_column].tolist()

    else:
        raise Exception("invalid mode")

    train_dataset = TensorDataset(
        features_embeds_train, torch.tensor(targets_train, dtype=torch.float32)
    )

    # tabGPT model
    model_config = tabGPT.get_default_config()
    model_config.model_type = "gpt-micro"
    model_config.vocab_size = 50257  # openai's model vocabulary
    model_config.block_size = max_length # 1024 is openai's model block_size
    model_config.n_output_nodes = 1
    model = tabGPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 100000
    train_config.epochs = 86 # used in individual comparison for cross-training of concept paper
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

    embed()

    test_dataset_sim_low = TensorDataset(
        Embedder(simulated_demand_low, mode='test').embed(n_cols=n_cols),
        torch.tensor(simulated_demand_low.df_test[target_column].tolist(), dtype=torch.float32),
    )

    test_dataset_sim_high = TensorDataset(
        Embedder(simulated_demand_high, mode='test').embed(n_cols=n_cols),
        torch.tensor(simulated_demand_high.df_test[target_column].tolist(), dtype=torch.float32),
    )

    df_test_sim_low = predict(
        model, DataLoader(test_dataset_sim_low, batch_size=32), simulated_demand_low.df_test
    )
    evaluation(simulated_demand_low.df_test[target_column], df_test_sim_low["yhat"])
    plot_timeseries(df_test_sim_low, "val_low", True)
    for pg in df_test_sim_low["product group id"].unique():
        plot_timeseries(df_test_sim_low[df_test_sim_low["product group id"] == pg], str(pg) + "_val_low", True)

    df_test_sim_high = predict(
        model, DataLoader(test_dataset_sim_high, batch_size=32), simulated_demand_high.df_test
    )
    evaluation(simulated_demand_high.df_test[target_column], df_test_sim_high["yhat"])
    plot_timeseries(df_test_sim_high, "val_high", True)
    for pg in df_test_sim_high["product group id"].unique():
        plot_timeseries(df_test_sim_high[df_test_sim_high["product group id"] == pg], str(pg) + "_val_high", True)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
