import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import Accuracy, AUROC
from model import ModifiedMobileNetV3_Small

class ParkingSpotClassifier(pl.LightningModule):
    def __init__(self):
        super(ParkingSpotClassifier, self).__init__()
        self.model = ModifiedMobileNetV3_Small(num_classes=1)
        self.criterion = nn.BCEWithLogitsLoss()

        # metrics
        self.accuracy = Accuracy(task="binary")
        self.auc_score = AUROC(task="binary")

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        int_list = [int(element) for element in labels]
        labels_tensor = torch.tensor(int_list)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            images = images.to(device)
            labels_tensor = labels_tensor.to(device)

        outputs = self.forward(images).squeeze()
        loss = self.criterion(outputs, labels_tensor.float())
        self.log('train_loss', loss)

        preds = torch.sigmoid(outputs) > 0.5

        acc = self.accuracy(preds, labels_tensor)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True)
        auc = self.auc_score(outputs, labels_tensor)
        self.log('train_auc_score', auc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        int_list = [int(element) for element in labels]
        labels_tensor = torch.tensor(int_list)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            images = images.to(device)
            labels_tensor = labels_tensor.to(device)

        outputs = self.forward(images).squeeze()
        loss = self.criterion(outputs, labels_tensor.float())
        self.log('val_loss', loss)

        preds = torch.sigmoid(outputs) > 0.5

        acc = self.accuracy(preds, labels_tensor)
        self.log('validation_accuracy', acc, on_step=True, on_epoch=True)
        auc = self.auc_score(outputs, labels_tensor)
        self.log('validation_auc_score', auc, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch

        int_list = [int(element) for element in labels]
        labels_tensor = torch.tensor(int_list)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            images = images.to(device)
            labels_tensor = labels_tensor.to(device)

        outputs = self.forward(images).squeeze()
        loss = self.criterion(outputs, labels_tensor.float())
        self.log('test_loss', loss)

        preds = torch.sigmoid(outputs) > 0.5

        acc = self.accuracy(preds, labels_tensor)
        self.log('test_accuracy', acc, on_step=True, on_epoch=True)
        auc = self.auc_score(outputs, labels_tensor)
        self.log('test_auc_score', auc, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0005)
