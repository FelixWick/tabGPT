from tabgpt.trainer import Trainer
from torch.utils.data.dataloader import DataLoader
import torch

def whole_epoch_train_loss(trainer: Trainer):
    aggregated_loss = 0
    iter_in_epoch = 0
    trainer.model.eval()
    for x, y in DataLoader(trainer.train_dataset, batch_size=32):
        with torch.no_grad():
            _, loss = trainer.model(x.to(trainer.device), y.to(trainer.device))
        aggregated_loss += loss.item()
        iter_in_epoch += 1
    trainer.aggregated_loss = aggregated_loss/iter_in_epoch