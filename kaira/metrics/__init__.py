from typing_extensions import Literal
from pytorch_msssim import ms_ssim
from torch import Tensor
import torch
import torchmetrics
from torchmetrics import MeanMetric
from typing import Any
from torchmetrics.image.inception import InceptionScore, Tuple
from torchmetrics.functional.image.lpips import _lpips_compute, _lpips_update

# A metric class that computes the multi-scale structural similarity index measure (SSIM) between two images.
class MultiScaleSSIM(MeanMetric):

    def __init__(self, kernel_size=11, data_range=1.0, **kwargs: Any) -> None:
        '''The function initializes with specified kernel size and data range parameters.
        
        Parameters
        ----------
        kernel_size, optional
            The `kernel_size` parameter is used to specify the size of the kernel or filter in a
        convolutional neural network. It determines the receptive field of the filter, which is the area
        of the input image that the filter is applied to. A larger kernel size will result in a larger
        receptive field and
        data_range
            The `data_range` parameter represents the range of the input data. It is used to normalize the
        input data before applying any operations. The default value is 1.0, which means that the input
        data is assumed to be in the range [0, 1].
         : Any
            - `kernel_size`: The size of the kernel used for convolutional operations. It determines the
        size of the receptive field.
        
        '''
        super().__init__("warn", **kwargs)

        self.kernel_size = kernel_size
        self.data_range = data_range
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        value = ms_ssim(preds, targets, data_range=1.0, size_average=False, win_size=self.kernel_size)

        return super().update(value, 1)

class LearnedPerceptualImagePatchSimilarity(torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity):
    
    def __init__(self, net_type: Literal['vgg', 'alex', 'squeeze'] = "alex", normalize: bool = False, **kwargs: Any) -> None:
        '''The function initializes a class instance with specified parameters and adds a state variable.
        
        Parameters
        ----------
        net_type : Literal['vgg', 'alex', 'squeeze'], optional
            The `net_type` parameter is a string that specifies the type of neural network to be used. It
        can take one of three values: 'vgg', 'alex', or 'squeeze'.
        normalize : bool, optional
            The `normalize` parameter is a boolean flag that indicates whether or not to normalize the
        input data. If `normalize` is set to `True`, the input data will be normalized before being
        passed through the network. If `normalize` is set to `False`, the input data will not be
        normalized
         : Any
            - `net_type`: A string literal specifying the type of neural network. It can be one of 'vgg',
        'alex', or 'squeeze'. The default value is "alex".
        
        '''
        super().__init__(net_type, normalize=normalize, reduction="mean", **kwargs)
        
        self.add_state("sum_sq", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:
        '''The `update` function calculates the LPIPS score between two images and updates the internal
        states.
        
        Parameters
        ----------
        img1 : Tensor
            Tensor representing the first image.
        img2 : Tensor
            Tensor - The second image for calculating the LPIPS score.
        
        '''
        loss, total = _lpips_update(img1, img2, net=self.net, normalize=self.normalize)
        
        self.sum_scores += loss.sum()
        self.total += total
        
        self.sum_sq += (loss ** 2).sum()

    def compute(self) -> Tensor:
        '''The function computes the final perceptual similarity metric by calculating the mean and
        standard deviation of the scores.
        
        Returns
        -------
            two values: `mean` and `std`.
        
        '''
        mean = _lpips_compute(self.sum_scores, self.total, "mean")
        std = torch.sqrt((self.sum_sq / self.total) - mean**2)
        
        return mean, std
    
class PeakSignalNoiseRatio(torchmetrics.image.PeakSignalNoiseRatio):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reduction=None, dim=[1,2,3], **kwargs)

    def compute(self):
        '''The compute function calculates the mean and standard deviation of the results per sample.
        
        Returns
        -------
            The compute() method returns the mean and standard deviation of the res_per_sample variable.
        
        '''
        res_per_sample = super().compute()
        return res_per_sample.mean(), res_per_sample.std()

# A metric class that computes the structural similarity index measure (SSIM) between two images.
class StructuralSimilarityIndexMeasure(torchmetrics.image.StructuralSimilarityIndexMeasure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reduction=None, **kwargs)

    def compute(self):
        '''The function computes the mean and standard deviation of a set of samples.
        
        Returns
        -------
            the mean and standard deviation of the metric
        '''
        res_per_sample = super().compute()
        
        return res_per_sample.mean(), res_per_sample.std()
