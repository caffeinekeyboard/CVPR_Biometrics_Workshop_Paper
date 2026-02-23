import torch
import torch.nn as nn


class matcher(nn.Module):
    def __init__(self, model1: nn.Module, model2: nn.Module, metric, performance, mask: bool = False):
        super().__init__()
        self.alignment = model1
        self.extractor = model2
        self.metric = metric
        self.performance = performance
        self.mask = mask
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        alignment1 = self.alignment(img1, img2)

        extractor1 = self.extractor(alignment1)
        extractor2 = self.extractor(img2)

        score = self.metric(alignment1, img2, extractor1, extractor2, mask=self.mask)
        
        performance = self.performance(score)
        return performance