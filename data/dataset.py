import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from datasets import load_dataset


class MultiNLIDataset(Dataset):

    def __init__(self, split: str, tokenizer: AutoTokenizer, do_tokenize: bool = True, do_prompt: bool = True) -> None:
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("multi_nli", split=self.split)
        self.columns_to_keep = set(['input_ids', 'attention_mask', 'label'])

        if do_prompt:
            self.__prepare_prompt()
        else:
            self.__prepare_inputs()

        if do_tokenize:
            self.__tokenize()
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        return self.data.__repr__()
    
    def __prepare_prompt(self) -> None:
        def prompt_function(examples):
            input_text = [
                'instruction: {} {} premise: {} {} hypothesis: {}'.format(
                    'Does the premise entaills the hypothesis?', self.tokenizer.eos_token, premise, self.tokenizer.eos_token, hypothesis
                )
                for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
            ]
            return dict(input_text=input_text)
        self.data = self.data.map(
            prompt_function, batched=True, load_from_cache_file=False,
            desc='Preparing prompts for MNLI {} dataset'.format(self.split)
        )

    def __prepare_inputs(self) -> None:
        def inputs_function(examples):
            input_text = [
                '{} {} {}'.format(premise, self.tokenizer.sep_token, hypothesis)
                for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
            ]
            return dict(input_text=input_text)
        self.data = self.data.map(
            inputs_function, batched=True, load_from_cache_file=False,
            desc='Preparing inputs for MNLI {} dataset'.format(self.split)
        )
    
    def __tokenize(self):
        preprocess_function = lambda examples: self.tokenizer(examples['input_text'], truncation=True)
        tokenized_data = self.data.map(
            preprocess_function, batched=True, load_from_cache_file=False,
            desc='Running tokenizer on MNLI {} dataset'.format(self.split)
        )
        print(tokenized_data[0])
        self.data = tokenized_data.remove_columns(set(tokenized_data.column_names) - self.columns_to_keep)
    
    def __getitem__(self, index) -> dict:
        return self.data[index]

class DummyDataset(Dataset):

    def __init__(self, split: str, tokenizer: AutoTokenizer, n: int, l: int) -> None:
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.data = [
            dict(
                input_ids=torch.ones(l).long(),
                attention_mask=torch.ones(l).long(),
                label=1
            ) for _ in range(n)
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        return self.data.__repr__()
    
    def __getitem__(self, index) -> dict:
        return self.data[index]