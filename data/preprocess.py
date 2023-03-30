class PaddingCollator():

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def __call__(self, inputs_list: list) -> dict:
        pad_inputs = self.tokenizer.pad(inputs_list, return_tensors='pt')
        inputs_dict = dict(input_ids=pad_inputs['input_ids'], attention_mask=pad_inputs['attention_mask'], label=pad_inputs['label'])
        return inputs_dict