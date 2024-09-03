import numpy as np
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import root_mean_squared_log_error
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2Config

import torch





class TabGPT(L.LightningModule):
    def __init__(self, model_path, tok_path, lr, pretrained=False):
        super().__init__()
        self.save_hyperparameters()
        self.col_embeddings = None
        self.validation_step_outputs = []
        self.predict_step_outputs = []
        self.lr = lr
        self.config = GPT2Config(device_map='auto',num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path,device_map='auto',num_labels=1)
        else:
            self.model = AutoModelForSequenceClassification.from_config(self.config)
            
        self.model.train()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def training_step(self, batch, batch_idx):
        input_embeds, target = batch
        out = self.model(inputs_embeds=input_embeds,labels=target)
        loss = out['loss']
        self.log("loss", loss, prog_bar=True, on_step=True,on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_embeds, target = batch
        out = self.model(inputs_embeds=input_embeds,labels=target)
        val_loss = out['loss']
        self.validation_step_outputs.append((out['logits'], target))
        self.log("val_loss", val_loss, prog_bar=True, on_step=True,on_epoch=True)

        return val_loss
    
    def on_validation_epoch_end(self):
        all_results =[list(t) for t in zip(*self.validation_step_outputs)]
        all_preds = torch.cat(all_results[0],dim=0).view(-1).cpu().float().numpy()
        all_labels = torch.cat(all_results[1],dim=0).view(-1).cpu().float().numpy()

        if np.all(all_preds > 0):
            rmsle = root_mean_squared_log_error(np.exp(all_preds),np.exp(all_labels))
            self.log('rmsle',rmsle,prog_bar=True,on_step=False,on_epoch=True)
        self.validation_step_outputs.clear()  # free memory

    def predict_step(self, batch, batch_idx): 
        input_embeds, target = batch
        pred = self.model(inputs_embeds=input_embeds)['logits']
        return np.exp(pred.cpu().float().numpy())
   
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(self.trainer.fit_loop._data_source.instance),  # Total number of steps over the entire training
            pct_start=0.3,  # Optional: Adjust this based on your needs
            anneal_strategy='cos',  # Optional: 'cos' or 'linear'
            cycle_momentum=False  # Optional: Use this if using AdamW or similar optimizer
        )

        # Return both optimizer and scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler}