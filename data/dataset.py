import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from datasets import load_dataset, Value
from data.prompt import BasePromptClass

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


class MNLIDataset(Dataset):

    def __init__(self, split: str, tokenizer: AutoTokenizer = None, prompt: BasePromptClass = None, to_binary: bool = False) -> None:
        super().__init__()
        self.data_name = 'multi_nli'
        self.split = split
        self.tokenizer = tokenizer
        self.prompt = prompt

        # Load data
        self.__load_data()
        # Build prompts
        self.__build_prompts()

        assert 'input_text' in self.data.column_names, 'column `input_text` is missing'

        print('Data format: \n\t{}'.format(
            '\n\t'.join([f'{k}: {v}' for k,v in self.data[0].items()])
        ))
        
        if to_binary:
            self.__convert_to_binary()
    
    def __load_data(self) -> None:
        self.data = load_dataset("multi_nli", split=self.split)

    def __build_prompts(self) -> None:
        prompt_fn = self.prompt.prompt
        self.data = self.data.map(
            prompt_fn, batched=True, load_from_cache_file=False, fn_kwargs=dict(tokenizer=self.tokenizer),
            desc='Preparing prompts for `{}` dataset ({} split)'.format(self.data_name, self.split)
        )
    
    def __convert_to_binary(self) -> None:
        self.data = self.data.map(
            lambda examples: {**examples, **{'label': [0.0 if x > 0 else 1.0 for x in examples['label']]}},
            batched=True, load_from_cache_file=False,
            desc='Convert `{}` dataset ({} split) to binary'.format(self.data_name, self.split)
        ).cast_column('label', Value('float32'))

    def __getitem__(self, index) -> dict:
        return self.data[index]
    
    def __len__(self) -> int :
        return len(self.data)
    
    def __repr__(self) -> str:
        return self.data.__repr__()


class ZeroDataset(Dataset):

    def __init__(self, split: str, files: list[str], tokenizer: AutoTokenizer = None, prompt: BasePromptClass = None, to_binary: bool = False) -> None:
        super().__init__()
        self.data_name = files
        self.split = split
        self.tokenizer = tokenizer
        self.prompt = prompt

        # Load data
        self.__load_data()
        # Build prompts
        self.__build_prompts()

        assert 'input_text' in self.data.column_names, 'column `input_text` is missing'
        assert 'group' in self.data.column_names, 'column `group` is missing'

        print('Data format: \n\t{}'.format(
            '\n\t'.join([f'{k}: {v}' for k,v in self.data[0].items()])
        ))

        if to_binary:
            self.__convert_to_binary()
    
    def __load_data(self) -> None:
        self.data = load_dataset('json', data_files=self.data_name, split='train')

    def __build_prompts(self) -> None:
        prompt_fn = self.prompt.prompt
        self.data = self.data.map(
            prompt_fn, with_indices=True, batched=True, load_from_cache_file=False, fn_kwargs=dict(tokenizer=self.tokenizer),
            desc='Preparing prompts for `{}` dataset ({} split)'.format(self.data_name, self.split),
            remove_columns=self.data.column_names
        )
    
    def __convert_to_binary(self) -> None:
        self.data = self.data.map(
            lambda examples: examples.update({'label': [1 if x == 0 else 0 for x in examples['label']]}),
            batched=True, load_from_cache_file=False,
            desc='Convert `{}` dataset ({} split) to binary'.format(self.data_name, self.split)
        ).cast_column('label', Value('float32'))

    def __getitem__(self, index) -> dict:
        return self.data[index]
    
    def __len__(self) -> int :
        return len(self.data)
    
    def __repr__(self) -> str:
        return self.data.__repr__()
