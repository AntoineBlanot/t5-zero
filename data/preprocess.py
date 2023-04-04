class PaddingCollator():

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def __call__(self, inputs_list: list) -> dict:
        pad_inputs = self.tokenizer.pad(inputs_list, return_tensors='pt')
        inputs_dict = dict(input_ids=pad_inputs['input_ids'], attention_mask=pad_inputs['attention_mask'], label=pad_inputs['label'])
        return inputs_dict
    

class ZeroShotPaddingCollator():

    def __init__(self, tokenizer, metadata_col: list[str] = []) -> None:
        self.tokenizer = tokenizer
        self.metadata_col = metadata_col
    
    def __call__(self, inputs_list: list) -> dict:
        """
        Pad the inputs and retrieve metadata in a separate dictionary
        """
        metadata_dict = {k: [x.pop(k) for x in inputs_list] for k in self.metadata_col}
        
        pad_inputs = self.tokenizer.pad(inputs_list, return_tensors='pt')
        inputs_dict = dict(input_ids=pad_inputs['input_ids'], attention_mask=pad_inputs['attention_mask'], label=pad_inputs['label'])
        
        res_dict = dict(inputs=inputs_dict, metadata=metadata_dict)
        return res_dict