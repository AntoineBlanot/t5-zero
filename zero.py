import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from tqdm import tqdm


class ZeroShotPredictor():

    def __init__(self,
        model: nn.Module, data_loader: DataLoader, device: str = 'cpu', compute_metrics: callable = None
    ) -> None:
        self.model = model.to(device)
        self.data_loader = data_loader
        self.compute_metrics = compute_metrics

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

                all_outputs.append(outputs['head_outputs'].detach())
                all_labels.append(labels)
                all_metadata.append(metadata)

        metadata_dict = {k: sum([x[k] for x in all_metadata], []) for k in all_metadata[0].keys()}
        outputs_dict = dict(outputs=all_outputs, labels=all_labels)
        outputs_dict = {k: torch.cat(v) for k,v in outputs_dict.items()}

        res_dict = {**outputs_dict, **metadata_dict}

        if self.compute_metrics is not None:
            metrics = self.compute_metrics(res_dict)
        else:
            metrics = {}

        return metrics

