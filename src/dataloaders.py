""" 
    Slight modifications to GLUEDataModule from:
    https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html 
"""


import datasets
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer


class GLUEDataModule(LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    glue_task_str_output = {
        "cola": False,
        "sst2": False,
        "mrpc": False,
        "qqp": False,
        "stsb": False,
        "mnli": True,
        "qnli": True,
        "rte": True,
        "wnli": False,
        "ax": True,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        test_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)
        self.label_mapping = {}

        sorted_keys = sorted(self.dataset.keys())
        for split in sorted_keys:
            self.label_mapping[split] = self.dataset[split].features['label']
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

            # TODO: REMOVE ME
            #if split == 'train':
            #    self.dataset[split] = self.dataset[split].filter(lambda x, idx: idx < 128, with_indices=True)

            # BK: Reset test labels to 0's so the model won't throw an error
            def zero_labels(row):
                row['labels'] = max(0, row['labels'])
                return row

            if 'test' in split:
                self.dataset[split] = self.dataset[split].map(zero_labels)
                self.test_labels = self.dataset['train'].features['label']

        self.eval_splits = [x for x in sorted_keys if "validation" in x]
        self.test_splits = [x for x in sorted_keys if "test" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.test_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.test_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.test_batch_size) for x in self.test_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding='max_length', truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features
    

class BertDataModule(LightningDataModule):
    def __init__(self, 
                 tokenizer_name: str,
                 max_seq_length: int = 128,
                 train_batch_size: int = 32,
                 **kwargs):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def prepare_data(self):
        # called only on 1 GPU

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()
    
    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding='max_length', truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


def test():
    pass