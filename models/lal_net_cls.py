import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F


class LalNetClsCE(pl.LightningModule):
    def __init__(self,
                 input_shape=406,
                 view=False,
                 weights_tensor=torch.Tensor([1, 4]),
                 device_use='cuda'
                 ):
        super(LalNetClsCE, self).__init__()
        self.device_use = device_use
        self._view = view
        self.weights_tensor = weights_tensor.to(self.device_use)
        self.input_shape = input_shape

        self.encode = nn.Sequential(
            nn.Linear(self.input_shape, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),

            nn.Linear(32, 32)
        )
        self.cls_linear = nn.Linear(32, 2)
        self.activation = nn.Softmax()

    def forward(self, x):
        embedded = self.encode(x)
        return self.activation(self.cls_linear(embedded))

    def vectorize(self, x):
        return self.encode(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.to(self.device_use)
        y = y.to(self.device_use)
        if self._view:
            x = x.view(x.size(0), -1)
        z = self.encode(x)
        activation_z = self.activation(self.cls_linear(z))
        loss = F.cross_entropy(activation_z, y.long(), weight=self.weights_tensor)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.to(self.device_use)
        y = y.to(self.device_use)
        if self._view:
            x = x.view(x.size(0), -1)
        z = self.encode(x)
        activation_z = self.activation(self.cls_linear(z))
        loss = F.cross_entropy(activation_z, y.long())
        self.log('val_loss', loss)


# class LalNetClsCE89(pl.LightningModule):
#     def __init__(self,
#                 input_shape=89,
#                 view=False,
#                 weights_tensor=torch.Tensor([1, 4])
#                 ):
#         super(LalNetClsCE, self).__init__()
#
#         self._view = view
#         self.weights_tensor = weights_tensor
#         self.input_shape = input_shape
#
#         self.encode = nn.Sequential(
#             nn.Linear(self.input_shape, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.BatchNorm1d(32),
#             nn.Dropout(0.1),
#
#             nn.Linear(32, 32)
#         )
#         self.cls_linear = nn.Linear(32, 2)
#         self.activation = nn.Softmax()
#
#
#     def forward(self, x):
#         embedded = self.encode(x)
#         return self.activation(self.cls_linear(embedded))
#
#     def vectorize(self, x):
#         return self.encode(x)
#
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
#         return optimizer
#
#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         if self._view:
#             x = x.view(x.size(0), -1)
#         z = self.encode(x)
#         activation_z = self.activation(self.cls_linear(z))
#         loss = F.cross_entropy(activation_z, y.long(), weight=self.weights_tensor)
#         self.log('train_loss', loss)
#         return loss
#
#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         if self._view:
#             x = x.view(x.size(0), -1)
#         z = self.encode(x)
#         activation_z = self.activation(self.cls_linear(z))
#         loss = F.cross_entropy(activation_z, y.long())
#         self.log('val_loss', loss)
