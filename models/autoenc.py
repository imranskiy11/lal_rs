import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class IndustryEmbAutoEncoder(pl.LightningModule):
    def __init__(self,
                 encoder_input_size=2144,
                 encoder_hidden_size=64,
                 device_use='cuda'):
        super().__init__()
        self.encoder_input_size = encoder_input_size
        self.encoder_hidden_size = encoder_hidden_size

        self.device_use = device_use

        self.encode = nn.Sequential(
            nn.Linear(self.encoder_input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(256, self.encoder_hidden_size),
            nn.BatchNorm1d(self.encoder_hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size))

        self.decode = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, self.encoder_input_size),
            nn.BatchNorm1d(self.encoder_input_size),
            nn.ReLU())


    def training_step(self, train_batch, batch_idx):
        print(f'len  train : {len(train_batch)}')
        # x = next(iter(train_batch))
        x, ctns = train_batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device_use)
        ctns = ctns.to(self.device_use)
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        print(f'len  val : {len(val_batch)}')
        # x = next(iter(val_batch))
        x, ctns = val_batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device_use)
        ctns = ctns.to(self.device_use)
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        return optimizer


class RawEmbAutoEncoder(pl.LightningModule):
    def __init__(self,
                 encoder_input_size=323,
                 encoder_hidden_size=64,
                 device_use='cuda'):
        super().__init__()
        self.encoder_input_size = encoder_input_size
        self.encoder_hidden_size = encoder_hidden_size
        self.device_use = device_use

        self.encode = nn.Sequential(
            nn.Linear(self.encoder_input_size, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.ReLU(),


            nn.Linear(128, self.encoder_hidden_size),
            nn.BatchNorm1d(self.encoder_hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size))

        self.decode = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, self.encoder_input_size),
            nn.BatchNorm1d(self.encoder_input_size),
            nn.ReLU())


    def training_step(self, train_batch, batch_idx):
        print(f'len  train : {len(train_batch)}')
        # x = next(iter(train_batch))
        x, ctns = train_batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device_use)
        ctns = ctns.to(self.device_use)
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        print(f'len  val : {len(val_batch)}')
        # x = next(iter(val_batch))
        x, ctns = val_batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device_use)
        ctns = ctns.to(self.device_use)
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        return optimizer
