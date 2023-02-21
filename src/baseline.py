import torch
import pytorch_lightning as pl

from typing import Callable

from sentence_transformers import SentenceTransformer
from torch import optim, nn
from torch.nn.functional import binary_cross_entropy
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy

from dataloaders import CoLADataModule


class EmbeddingClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, embedding_width:int, 
                 loss_fn: Callable, metric: Metric,
                 learning_rate: float):
        super().__init__()

        self.loss_fn = loss_fn
        self.metric = metric()
        self.learning_rate = learning_rate

        if n_classes == 2:
            self.network = nn.Sequential(nn.Linear(embedding_width, 1), nn.Sigmoid())
        else:
            self.network = nn.Sequential(nn.Linear(embedding_width, n_classes), nn.Softmax())

    def forward(self, x):
        y = self.network(x)
        return y

    def training_step(self, batch, batch_idx):
        x, targets = batch
        y = self(x)
        loss = self.loss_fn(y, targets)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, targets = batch
        y = self(x)
        val_loss = self.loss_fn(y, targets)
        self.log("val_loss", val_loss)
        self.log("test_accuracy", self.metric(y, targets))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def run_baseline(cfg):
    sentence_encoder = SentenceTransformer('all-mpnet-base-v2')

    batch_size = 1000
    for task in cfg.tasks:
        print(task)
        print(cfg[task])
        dm = globals()[cfg[task].datamodule](data_dir=cfg[task].data_dir, 
                                             batch_size=batch_size,
                                             sentence_encoder=sentence_encoder)

        cls = EmbeddingClassifier(n_classes=cfg[task].n_classes, 
                                  embedding_width=cfg[task].embedding_width,
                                  loss_fn=globals()[cfg[task].loss_fn],
                                  metric=globals()[cfg[task].metric],
                                  learning_rate=cfg[task].learning_rate)

        dm.setup(stage='fit')
        dm.setup(stage='validate')
        dm.setup(stage='test')

        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=20)
        trainer.fit(model=cls, datamodule=dm)
        trainer.predict(model=cls, datamodule=dm)