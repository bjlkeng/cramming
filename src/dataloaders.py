import os

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional


class GlueDataset(Dataset):
 
    def __init__(self, filename: str, target_index: int, 
                 s1_index: int , s2_index: int, test_index: int = None,
                 header: Optional[int] = None):
        self.target_index = target_index
        self.s1_index = s1_index
        self.s2_index = s2_index
        self.test_index = test_index

        df_glue = pd.read_csv(filename, sep='\t', header=header)
        col_indexes = [col for col in [s1_index, s2_index, test_index] if col is not None]
        self.y_train = df_glue.iloc[:, target_index].values
        self.x_train = df_glue.iloc[:, col_indexes].values
 
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]


class GlueBaseDataModule(pl.LightningDataModule):
    pass

    @staticmethod
    def parse_tsv(filename: str, target_index: int, s1_index: int , s2_index: int):

        return data


class CoLADataModule(GlueBaseDataModule):
    def __init__(self, data_dir: str = './glue_data/CoLA/'):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit':
            self.train = GlueDataset(filename=os.path.join(self.data_dir, 'train.tsv'),
                                     target_index=1, s1_index=3, s2_index=None)
            
        if stage == 'validate':
            self.val = GlueDataset(filename=os.path.join(self.data_dir, 'dev.tsv'),
                                   target_index=1, s1_index=3, s2_index=None)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test':
            self.test = GlueDataset(filename=os.path.join(self.data_dir, 'test.tsv'),
                                    target_index=0, s1_index=1, s2_index=None,
                                    header=0)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)



if __name__ == "__main__":
    data_modules = [CoLADataModule()]

    for dm in data_modules:
        for stage in ['fit', 'validate', 'test']:
            dm.setup(stage=stage)

            if stage == 'fit':
                dataset = dm.train
            elif stage == 'validate':
                dataset = dm.val
            elif stage == 'test':
                dataset = dm.test
            else:
                assert False, stage

            print(stage)
            for i, (x, y) in enumerate(dataset):
                if i < 10:
                    print(f'i: x = {x}, y = {y}')