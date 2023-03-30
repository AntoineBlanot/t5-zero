from transformers import AutoTokenizer
from model.modeling import MyModel

name = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(name)
model = MyModel(name)
print('Model params: {} M'.format(int(sum(p.numel() for p in model.parameters())/1e6)))
print('Trainable params: {} M'.format(int(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)))

input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt", padding=True).input_ids
print(input_ids.shape)

outputs_dict = model(input_ids=input_ids)
print({k: v.shape for k,v in outputs_dict.items()})
