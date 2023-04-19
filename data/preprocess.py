from typing import Dict, List

import torch


class TokenizeAndPad():

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def __call__(self, inputs_list: List) -> Dict[str, torch.Tensor]:
        """
        Tokenize and pad the inputs
        """
        input_text = [x['input_text'] for x in inputs_list]
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        
        labels = torch.as_tensor([x['label'] for x in inputs_list])

        return dict(
            **inputs,
            label=labels
        )


class BARTTokenizeAndPad():
    """
    Data Collator for BART. BART is special because it each example in the batch needs to have the same number of eos token (because of truncation we have to be careful)
    """
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def __call__(self, inputs_list: List) -> Dict[str, torch.Tensor]:
        """
        Tokenize and pad the inputs
        """
        input_text = [x['input_text'] for x in inputs_list]
        a, b = zip(*[[x for x in t.split(self.tokenizer.eos_token) if x != ''] for t in input_text])
        inputs = self.tokenizer(list(a), list(b), return_tensors='pt', padding=True, truncation='only_first')

        labels = torch.as_tensor([x['label'] for x in inputs_list])

        return dict(
            **inputs,
            label=labels
        )
    

class ZeroCollator(TokenizeAndPad):

    def __init__(self, tokenizer, metadata_columns: List[str] = None) -> None:
        super().__init__(tokenizer=tokenizer)
        self.metadata_columns = metadata_columns if metadata_columns is not None else []
    
    def __call__(self, inputs_list: List) -> Dict[str, torch.Tensor]:
        """
        Deal with metadata separately. Then Tokenize and Pad inputs.
        """
        # Extract metadata
        metadata_dict = {k: [x.pop(k) for x in inputs_list] for k in self.metadata_columns}
        # Tokenize and Pad inputs
        inputs_dict = super().__call__(inputs_list=inputs_list)

        return dict(
            inputs=inputs_dict,
            metadata=metadata_dict
        )