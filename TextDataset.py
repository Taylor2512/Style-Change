 
import torch
import torch.nn as nn
 
from torch.utils.data import Dataset
 
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text1 = str(self.texts[item][0])
        text2 = str(self.texts[item][1])
        label = self.labels[item]

        encoding = self.tokenizer(
            text1,
            text2,
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )

        return {
            'position': item,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels':  torch.tensor([1 - label, label])
        }
     
 