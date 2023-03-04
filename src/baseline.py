from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import datasets
from omegaconf import ListConfig
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from dataloaders import GLUEDataModule
from utils import get_hydra_output_dir


class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            'glue', self.hparams.task_name, experiment_id=datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch['labels']

        return {'loss': val_loss, 'preds': preds, 'labels': labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == 'mnli':
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split('_')[-1]
                preds = torch.cat([x['preds'] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x['labels'] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x['loss'] for x in output]).mean()
                self.log(f'val_loss_{split}', loss, prog_bar=True)
                split_metrics = {
                    f'{k}_{split}': v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        return preds

    def configure_optimizers(self):
        '''Prepare optimizer and schedule (linear warmup and decay)'''
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]



def run_baseline(cfg):

    batch_size = cfg.batch_size
    model_name = cfg.pretrained_model
    seed = cfg.seed
    
    for task in cfg.tasks:
        wandb_logger = WandbLogger(project='cramming',
                                   name='baseline-' + task,
                                   log_model=False,
                                   save_dir=get_hydra_output_dir())
        wandb_logger.experiment.config.update(dict(cfg))

        seed_everything(seed)
        dm = GLUEDataModule(model_name_or_path=model_name, task_name=task, 
                            train_batch_size=batch_size,
                            eval_batch_size=batch_size)

        dm.setup('fit')
        model = GLUETransformer(
            model_name_or_path=model_name,
            num_labels=dm.num_labels,
            eval_splits=dm.eval_splits,
            task_name=dm.task_name,
        )

        # BK: Using val_loss to pick best model for simplicity here
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=cfg[task].metric_name)
        trainer = Trainer(
            max_epochs=cfg.max_epochs,
            accelerator='auto',
            devices=1 if torch.cuda.is_available() else None,
            default_root_dir=get_hydra_output_dir(),
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model, datamodule=dm)

        test_loaders = dm.test_dataloader()
        if type(test_loaders) is not list:
            test_loaders = [test_loaders]

        for i, loader in enumerate(test_loaders):
            predictions = trainer.predict(model, loader, ckpt_path='best')
            predictions = torch.concat(predictions)
            predictions = predictions.detach().cpu().numpy()

            df_predictions = dm.dataset[dm.test_splits[i]].select_columns('idx').to_pandas()
            df_predictions = df_predictions.rename(columns={'idx': 'id'})
            df_predictions['label'] = predictions

            filename = [cfg[task].outfile] if not isinstance(cfg[task].outfile, ListConfig) else cfg[task].outfile
            outpath = Path(get_hydra_output_dir()) / 'glue_outputs'
            outpath.mkdir(exist_ok=True)
            outpath = outpath / filename[i]
            df_predictions.to_csv(outpath, sep='\t', index=False)

        wandb_logger.finalize('success')