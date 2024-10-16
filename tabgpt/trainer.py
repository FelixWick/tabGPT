"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import root_mean_squared_log_error
from tabgpt.utils import CfgNode as CN
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
import os
import re
import logging



import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

    def __init__(self, config, model, train_dataset, log_dir='./logs', use_scheduler=True, target_scaler=None, progress_bar=True):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.use_scheduler = use_scheduler
        self.target_scaler = target_scaler
        self.disable_progress_bar = not progress_bar

        # Initialize the SummaryWriter

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.listdir(log_dir):
            logdir = os.path.join(log_dir,'version_0')
        else:
            folders = os.listdir(log_dir)
            max_version = max([int(re.search(r'\d+', folder).group()) for folder in folders])
            logdir = os.path.join(log_dir, f'version_{max_version+1}')
        self.writer = SummaryWriter(log_dir=logdir)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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

    def scheduler(self, optimizer, total_steps):
          return OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.3,  # Optional: Adjust this based on your needs
            anneal_strategy="cos",  # Optional: 'cos' or 'linear'
            cycle_momentum=True,  # Optional: Use this if using AdamW or similar optimizer
        )

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)
        #scheduler = ReduceLROnPlateauBest(self.optimizer, factor=0.5, patience=20)
        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        if self.use_scheduler:
            scheduler = self.scheduler(self.optimizer, total_steps=len(train_loader) * self.config.epochs)

        scaler = torch.amp.GradScaler()

        self.iter_num = 0
        self.epochs_run = 0
        self.iter_time = time.time()
        for epoch in range(config.epochs):
            self.aggregated_loss = 0
            self.iter_in_epoch = 0
            self.epochs_run += 1
            self.epoch_loss = 0
            self.epoch_rmsle = 0
            model.train()

            logging.info(f'Processed: {self.epochs_run}')

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}', position=0, leave=True, disable=self.disable_progress_bar)

            for batch_idx, batch in progress_bar:
                batch = [t.to(self.device) for t in batch]
                x, y = batch
                # backprop and update the parameters
                model.zero_grad(set_to_none=True)

                model.zero_grad(set_to_none=True)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    preds, self.loss = model(x=x, targets=y,return_dict=False)
        
                # add loss to TB
                self.writer.add_scalar('Loss/step_loss', self.loss.item(), self.iter_num)
                # compute scaled gradients
                scaler.scale(self.loss).backward()# self.loss.backward()
                # clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                # update params
                scaler.step(self.optimizer)

                if self.use_scheduler:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                    self.writer.add_scalar('Learning Rate', current_lr, self.iter_num)

                scaler.update()

                self.trigger_callbacks('on_batch_end')
                self.iter_num += 1
                self.iter_in_epoch += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                # Accumulate loss
                self.epoch_loss += self.loss.item()
        
                # Calculate average loss for the epoch so far
                avg_loss = self.epoch_loss / (batch_idx + 1)

                if self.target_scaler is not None:
                    orig_targ= self.target_scaler.inverse_transform(y.reshape(-1,1).cpu().numpy())
                    orig_pred = self.target_scaler.inverse_transform(preds.reshape(-1,1).detach().cpu().numpy())
                    self.epoch_rmsle += root_mean_squared_log_error(orig_targ, orig_pred)
                    avg_rmsle = self.epoch_rmsle / (batch_idx + 1)


                if self.iter_in_epoch == len(train_loader):
                    self.trigger_callbacks('on_epoch_end')
                    self.writer.add_scalar('Loss/avg_loss', avg_loss ,self.epochs_run)
                    if self.target_scaler is not None:
                        self.writer.add_scalar('Loss/avg_rmsle', avg_rmsle ,self.epochs_run)
                    self.writer.add_scalar('Loss/epoch_loss', self.aggregated_loss, self.epochs_run)



                progress_bar.set_postfix({'iter_dt': f'{self.iter_dt * 1000:.2f}ms',
                                         'loss': f'{self.loss.item():.3f}',
                                         'rmsle': f'{avg_rmsle:.3f}' if self.target_scaler is not None else 0,
                                         'Epoch_loss': self.aggregated_loss})
                
                # termination conditions
                if config.max_iters is not None and self.iter_num >= config.max_iters:
                    break
                # scheduler.step(self.aggregated_loss


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