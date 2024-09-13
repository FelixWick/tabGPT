"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from tabgpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.epochs = 1
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.observe_train_loss = False
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, betas=config.betas)

        scheduler = ReduceLROnPlateauBest(self.optimizer, factor=0.5, patience=20)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        self.iter_num = 0
        self.iter_time = time.time()
        for epoch in range(config.epochs):
            data_iter = iter(train_loader)
            self.aggregated_loss = 0
            self.iter_in_epoch = 0
            model.train()

            while True:
                # fetch the next batch (x, y) and re-init iterator if needed
                try:
                    batch = next(data_iter)
                    # if batch[1].size(dim=0) < config.batch_size:
                    #     break
                except StopIteration:
                    break
                batch = [t.to(self.device) for t in batch]
                x, y = batch

                # forward the model
                self.loss = model(inputs_embeds=x, labels=y).loss

                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                self.trigger_callbacks('on_batch_end')
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                # termination conditions
                if config.max_iters is not None and self.iter_num >= config.max_iters:
                    break

            if config.observe_train_loss:
                model.eval()
                for x, y in DataLoader(self.train_dataset, batch_size=32):
                    with torch.no_grad():
                        self.loss = model(inputs_embeds=x.to(self.device), labels=y.to(self.device)).loss
                    self.aggregated_loss += self.loss
                    self.iter_in_epoch += 1
                self.aggregated_loss /= self.iter_in_epoch
                self.epoch = epoch + 1
                self.trigger_callbacks('on_epoch_end')
                # scheduler.step(self.aggregated_loss)


class ReduceLROnPlateauBest(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def step(self, metrics):
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        self.num_bad_epochs_before = self.num_bad_epochs

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if (self.num_bad_epochs_before > self.patience) and (self.best == current):
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
