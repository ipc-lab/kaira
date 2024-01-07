from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MultiScaleSSIM

class MSELoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        '''The function initializes an instance of a class and sets a variable to an instance of the
        MSELoss class from the nn module.
        
        '''
        super().__init__(**kwargs)
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, targets):
        loss = self.mse_loss(preds, targets)
        return loss


class CombinedLoss(nn.Module):
    
    def __init__(self, losses, weights, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        
        for loss in losses:
            assert isinstance(loss, nn.Module)

    def forward(self, preds, targets):
        acc_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            acc_loss = acc_loss + weight * loss(preds, targets)

        return acc_loss

class MSELPIPSLoss(CombinedLoss):
    def __init__(self, lpips_weight: float, **kwargs) -> None:
        super().__init__(losses=[
            MSELoss(),
            LPIPSLoss(),
        ], weights=[1.0, lpips_weight], **kwargs)

class LPIPSLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True, reduction="mean"
        )
        self.lpips_loss.requires_grad_(False)

    def forward(self, preds, targets):
        loss = self.lpips_loss(preds.clip(0.0, 1.0), targets).mean()
        return loss

class SSIMLoss(nn.Module):
    def __init__(self, kernel_size=11, data_range=(0.0, 1.0), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.data_range = data_range
        self.loss = StructuralSimilarityIndexMeasure(kernel_size=self.kernel_size, data_range=self.data_range)
        
    def forward(self, preds, targets):
        loss = 1 - self.loss(preds, targets).mean()
        return loss
    
class MSSSIMLoss(nn.Module):
    def __init__(self, kernel_size=11, data_range=(0.0, 1.0), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.data_range = data_range
        self.loss = MultiScaleSSIM(kernel_size=self.kernel_size, data_range=self.data_range)
        
    def forward(self, preds, targets):
        loss = 1 - self.loss(preds, targets).mean()
        return loss

__all__ = ["MSELoss", "CombinedLoss", "MSELPIPSLoss", "LPIPSLoss", "SSIMLoss", "MSSSIMLoss"]