import torch
import pandas as pd
import lightning.pytorch as pl

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, config, data_path, has_labels):
        self.config = config
        self.data = pd.read_csv(data_path)
        self.has_labels = has_labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.lm.pretrained_esm)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]["Sequence"]
        tokens = self.tokenizer(
            sequence.upper(),
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.config.data.max_seq_len,
        )
        
        item = {
            "input_ids": tokens['input_ids'].squeeze(0),
            "attention_mask": tokens['attention_mask'].squeeze(0)
        }
        
        if self.has_labels:
            item["labels"] = torch.tensor(self.data.iloc[idx]["Label"], dtype=torch.float32)
        
        return item


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    masks = torch.stack([item['attention_mask'] for item in batch])
    batch_dict = {
        'input_ids': input_ids,
        'attention_mask': masks
    }
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        batch_dict['labels'] = labels
    return batch_dict


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, config, train_dataset, val_dataset, test_dataset, collate_fn=collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn
        self.batch_size = config.data.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collate_fn,
                          num_workers=8,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=8,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=8,
                          pin_memory=True)
