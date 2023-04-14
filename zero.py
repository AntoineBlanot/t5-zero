import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from tqdm import tqdm


class ZeroShotPredictor():

    def __init__(self,
        model: nn.Module, data_loader: DataLoader, device: str = 'cpu'
    ) -> None:
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
    
    def predict(self) -> dict:
        """
        Predict on the dataset
        """
        self.model.eval()
        pbar = tqdm(range(len(self.data_loader)), desc='Predicting')

        all_outputs, all_labels, all_metadata = [], [], []

        for i, data in enumerate(self.data_loader):
            with torch.no_grad():
                pbar.update()
                inputs, metadata = data['inputs'], data['metadata']
                inputs = {k: v.to(self.device) for k,v in inputs.items()}
                labels = inputs.pop('label')

                outputs = self.model(**inputs)
                logits = outputs['logits']

                all_outputs.append(logits.detach().cpu())
                all_labels.append(labels.cpu())
                all_metadata.append(metadata)

        metadata_dict = {k: sum([x[k] for x in all_metadata], []) for k in all_metadata[0].keys()}
        outputs_dict = dict(outputs=all_outputs, labels=all_labels)
        outputs_dict = {k: torch.cat(v) for k,v in outputs_dict.items()}

        res_dict = {**outputs_dict, **metadata_dict}

        metrics = self.model.compute_metrics(res_dict)

        return metrics

