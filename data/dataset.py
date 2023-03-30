import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from datasets import load_dataset


class MultiNLIDataset(Dataset):

    def __init__(self, split: str, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("multi_nli", split=self.split)
        self.columns_to_keep = set(['input_ids', 'attention_mask', 'label'])

        self.__tokenize()
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        return self.data.__repr__()
    
    def __tokenize(self):
        preprocess_function = lambda examples: self.tokenizer(examples['premise'], examples['hypothesis'], max_length=None)
        tokenized_data = self.data.map(
            preprocess_function, batched=True, load_from_cache_file=False,
            desc='Running tokenizer on MNLI {} dataset'.format(self.split)
        )
        self.data = tokenized_data.remove_columns(set(tokenized_data.column_names) - self.columns_to_keep)
    
    def __getitem__(self, index) -> dict:
        return self.data[index]
