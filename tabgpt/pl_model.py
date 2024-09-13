import logging
import os
from typing import Optional, Tuple, Union

import lightning as L
import numpy as np
from tabgpt.model_hf import tabGPT_HF
from tabgpt.tabular_dataset import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabgpt.model import tabGPT

import torch.optim as optim
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from transformers import (
    GPT2Config,
    BertConfig,
    GPT2Model,
    GPT2PreTrainedModel,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    SequenceClassifierOutputWithPast,
)

import torch
from safetensors.torch import save_file, load_file


from sklearn.metrics import root_mean_squared_log_error

from lightning.pytorch.callbacks import Callback
from transformers import AutoModelForSequenceClassification


from tabgpt.rank_loss import RnCLoss

logger = logging.getLogger(__name__)


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        pl_module.log("grad_norm", gradient_norm(pl_module))

def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class TabGPTPretrainer(L.LightningModule):
    def __init__(self, config, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = tabGPT_HF(config)
        self.model.train()
        self.lr = lr
        self.validation_step_outputs = []
        self.predict_step_outputs = []
       

    def training_step(self, batch, batch_idx):
        input_embeds, target = batch
        _, mse_loss = self.model(input_embeds, target)
        self.log("train_mse_loss", mse_loss, prog_bar=True, on_step=True, on_epoch=True)

        return mse_loss
    
    def validation_step(self, batch, batch_idx):
        input_embeds, target = batch
        tgt = torch.log(1 + target)
        preds, mse_loss = self.model(input_embeds, tgt)
        self.validation_step_outputs.append((preds, target))
        self.log("val_mse_loss", mse_loss, prog_bar=True, on_step=True, on_epoch=True)

        return mse_loss
    
    def on_validation_epoch_end(self):
        all_results =[list(t) for t in zip(*self.validation_step_outputs)]
        all_preds = torch.cat(all_results[0],dim=0).view(-1).cpu().float().numpy()
        all_labels = torch.cat(all_results[1],dim=0).view(-1).cpu().float().numpy()

        if np.all(all_preds > 0):
            rmsle = root_mean_squared_log_error(np.exp(all_preds)-1,all_labels)
            self.log('rmsle',rmsle,prog_bar=True,on_step=False,on_epoch=True)
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(
                self.trainer.fit_loop._data_source.instance
            ),  # Total number of steps over the entire training
            pct_start=0.3,  # Optional: Adjust this based on your needs
            anneal_strategy="cos",  # Optional: 'cos' or 'linear'
            cycle_momentum=True,  # Optional: Use this if using AdamW or similar optimizer
        )

        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
        #                                               base_lr=1e-6, 
        #                                               max_lr=1e-3, 
        #                                               step_size_up = 200, 
        #                                               step_size_down=400, 
        #                                               cycle_momentum=False)

        # scheduler = CosineAnnealingWarmRestarts(optimizer,
        #                                    T_0=100,
        #                                    T_mult=2,
        #                                    eta_min=1e-6)

        #scheduler = CosineAnnealingWarmRestarts(optimizer,
        #                                  T_0=80,
        #                                  T_mult=2,
        #                                  eta_min=1e-6)
        # Return both optimizer and scheduler
        scheduler = {"scheduler": scheduler,'interval': 'step', 'frequency': 1}
        return {"optimizer": optimizer, 'lr_scheduler': scheduler}



class TabGPTFinetuner(L.LightningModule):
    def __init__(self, config, lr):
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.predict_step_outputs = []
        self.lr = lr
        self.model = tabGPT_HF(config)
        self.model.train()
        self.lora_enabled = False


    def make_lora_model(self):
       
        all_linear = get_all_linear(self.model)
        # lm_head = all_linear[-1]
        peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=all_linear,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,  # Dropout rate
            )

        self.model = get_peft_model(self.model, peft_config)
        self.lora_enabled = True

    def training_step(self, batch, batch_idx):
        input_embeds, target = batch
        _, mse_loss = self.model(x=input_embeds, targets=target, return_dict=False)
        self.log("train_mse_loss", mse_loss, prog_bar=True, on_step=True, on_epoch=True)

        return mse_loss
    
    def validation_step(self, batch, batch_idx):
        input_embeds, target = batch
        tgt = torch.log(1 + target)
        preds, mse_loss = self.model(x=input_embeds, targets=tgt, return_dict=False)
        self.validation_step_outputs.append((preds, target))
        self.log("val_mse_loss", mse_loss, prog_bar=True, on_step=True, on_epoch=True)

        return mse_loss
    
    def on_validation_epoch_end(self):
        all_results =[list(t) for t in zip(*self.validation_step_outputs)]
        all_preds = torch.cat(all_results[0],dim=0).view(-1).cpu().float().numpy()
        all_labels = torch.cat(all_results[1],dim=0).view(-1).cpu().float().numpy()

        if np.all(all_preds > 0):
            rmsle = root_mean_squared_log_error(np.exp(all_preds)-1,all_labels)
            self.log('rmsle',rmsle,prog_bar=True,on_step=False,on_epoch=True)
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return {"optimizer": optimizer}


def save_weights(model, path, name='model'):
    
    # Get the model's state dict (weights)
    state_dict = model.state_dict()

    # Convert the state dict to a format compatible with safetensors
    # Safetensors expects a dictionary of tensors, so you don't need to change much here
    tensor_dict = {key: value for key, value in state_dict.items()}

    # Save the tensor dictionary as a safetensors file
    save_file(tensor_dict, f"{path}/{name}.safetensors")


def load_weights(model, path, name='model'):

    # Load the safetensors file
    loaded_tensors = load_file(f"{path}/{name}.safetensors")

    # Load the state dict into the model
    model.load_state_dict(loaded_tensors)
    return model


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