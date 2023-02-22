import os
import torch

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional


class GlueDataset(Dataset):
 
    def __init__(self, filename: str, target_index: int, 
                 s1_index: int , s2_index: int,
                 index_index: int = None,
                 header: Optional[int] = None,
                 sentence_encoder = None):
        self.target_index = target_index
        self.s1_index = s1_index
        self.s2_index = s2_index
        self.index_index = index_index

        df_glue = pd.read_csv(filename, sep='\t', header=header)
        col_indexes = [col for col in [s1_index, s2_index] if col is not None]
        self.x_train = df_glue.iloc[:, col_indexes].values.flatten()

        if target_index is not None:
            self.y_train = torch.tensor(df_glue.iloc[:, [target_index]].values, dtype=torch.float32)

        if index_index is not None:
            self.index = torch.tensor(df_glue.iloc[:, index_index].values)

        if sentence_encoder is not None:
            self.x_train = torch.tensor(sentence_encoder.encode(self.x_train))
        else:
            assert "No conversion to tensor available"
 
    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, index):
        if self.target_index is not None:
            return self.x_train[index], self.y_train[index]
        else:
            return self.x_train[index]



class GlueBaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, sentence_encoder=None, 
                 num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sentence_encoder = sentence_encoder
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers)


class CoLADataModule(GlueBaseDataModule):
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit':
            self.train = GlueDataset(filename=os.path.join(self.data_dir, 'train.tsv'),
                                     target_index=1, s1_index=3, s2_index=None,
                                     sentence_encoder=self.sentence_encoder)
            
        if stage == 'validate':
            self.val = GlueDataset(filename=os.path.join(self.data_dir, 'dev.tsv'),
                                   target_index=1, s1_index=3, s2_index=None,
                                   sentence_encoder=self.sentence_encoder)

        # Assign test dataset for use in dataloader(s)
        if stage == 'predict':
            self.predict = GlueDataset(filename=os.path.join(self.data_dir, 'test.tsv'),
                                       target_index=None, s1_index=1, s2_index=None, index_index=0,
                                       header=0, sentence_encoder=self.sentence_encoder)


def test():
    data_modules = [CoLADataModule(data_dir='./glue_data/CoLA/')]

    for dm in data_modules:
        for stage in ['fit', 'validate', 'predict']:
            dm.setup(stage=stage)

            if stage == 'fit':
                dataset = dm.train
            elif stage == 'validate':
                dataset = dm.val
            elif stage == 'test':
                dataset = dm.test
            elif stage == 'predict':
                dataset = dm.predict
            else:
                assert False, stage

            print('======================================')
            print(stage)
            for i, row in enumerate(dataset):
                if len(row) == 2:
                    x, y = row
                    if i < 10:
                        print(f'{i}: x = {x}, y = {y}')
                else:
                    x = row
                    print(f'{i}: x = {x}')