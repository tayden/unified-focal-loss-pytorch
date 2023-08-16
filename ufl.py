from __future__ import annotations

from abc import abstractmethod, ABC

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

EPSILON = 1e-10


class _Loss(nn.Module, ABC):
    ignore_index = None

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.
            
        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """
        raise NotImplementedError

    def _ignore_flatten(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """Flatten tensors and ignore pixels with ignore_index.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.


        Returns
        -------
        y_pred : Tensor of shape (num_samples - num_ignored_samples, num_classes)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (num_samples - num_ignored_samples, num_classes)
            Ground truth labels.
        """
        y_true = rearrange(y_true, "n c ... -> (n ...) c")
        y_pred = rearrange(y_pred, "n c ... -> (n ...) c")
        n, c = y_true.shape
        if self.ignore_index is not None:
            y_true = torch.argmax(y_true, dim=1).flatten()
            mask = y_true != self.ignore_index
            y_true = F.one_hot(y_true[mask], num_classes=c - 1)
            y_pred = y_pred[mask]

        return y_pred, y_true

    @staticmethod
    def _tfpn(
        y_pred, y_true, reduction="sum"
    ) -> (
        int | torch.Tensor,
        int | torch.Tensor,
        int | torch.Tensor,
        int | torch.Tensor,
    ):
        """Calculate true positives, true negatives, false positives and false negatives.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.
        reduction : str, optional
            Reduction method, by default "sum". If "none" is specified, the function
            returns the raw values.

        Returns
        -------
        tp : int | torch.Tensor
            Number of true positives.
        tn : int | torch.Tensor
            Number of true negatives.
        fp : int | torch.Tensor
            Number of false positives.
        fn : int | torch.Tensor
            Number of false negatives.

        """
        if not len(y_true.shape) == 2 or not len(y_pred.shape) == 2:
            raise ValueError("Metric: Shape of tensor is not 2D.")

        # Probs to one-hot
        n, c = y_true.shape
        y_pred = F.one_hot(y_pred.argmax(dim=1), num_classes=c)

        tp = y_true * y_pred
        tn = (1 - y_true) * (1 - y_pred)
        fn = y_true * (1 - y_pred)
        fp = (1 - y_true) * y_pred

        if reduction == "sum":
            return tp.sum(), tn.sum(), fp.sum(), fn.sum()
        elif reduction == "none":
            return tp, tn, fp, fn

        raise ValueError(f"Reduction {reduction} not supported.")


################################
#       Dice coefficient       #
################################
class DiceCoefficient(_Loss):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply
        Dice coefficient, is a statistical tool which measures the similarity between
        two sets of data.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def __init__(
        self, delta: float = 0.5, smooth: float = 0.000001, ignore_index: int = None
    ):
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        y_pred, y_true = self._ignore_flatten(y_pred, y_true)
        tp, tn, fp, fn = self._tfpn(y_pred, y_true)

        dice_class = (tp + self.smooth) / (
            (tp + self.delta * fn + (1 - self.delta) * fp) + self.smooth
        )
        # Average class scores
        return torch.mean(dice_class)


################################
#           Dice loss          #
################################
class DiceLoss(_Loss):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic
    developed in the 1940s to gauge the similarity between two samples.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def __init__(
        self, delta: float = 0.5, smooth: float = 0.000001, ignore_index: int = None
    ):
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        y_pred, y_true = self._ignore_flatten(y_pred, y_true)
        tp, tn, fp, fn = self._tfpn(y_pred, y_true)

        # Calculate Dice score
        dice_class = (tp + self.smooth) / (
            (tp + self.delta * fn + (1 - self.delta) * fp) + self.smooth
        )
        # Average class scores
        return torch.mean(1 - dice_class)


################################
#         Tversky loss         #
################################
class TverskyLoss(_Loss):
    """Tversky loss function for image segmentation using 3D fully convolutional deep
    networks. Link: https://arxiv.org/abs/1706.05721

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def __init__(
        self, delta: float = 0.7, smooth: float = 0.000001, ignore_index: int = None
    ):
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        y_pred, y_true = self._ignore_flatten(y_pred, y_true)
        tp, tn, fp, fn = self._tfpn(y_pred, y_true)

        tversky_class = (tp + self.smooth) / (
            (tp + self.delta * fn + (1 - self.delta) * fp) + self.smooth
        )
        # Average class scores
        return torch.mean(1 - tversky_class)


################################
#      Focal Tversky loss      #
################################
class FocalTverskyLoss(_Loss):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion
        segmentation
    Link: https://arxiv.org/abs/1810.07842

    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples,
        by default 0.75
    delta : float, optional
        controls weight given to each class, by default 0.6
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 0.000001,
        ignore_index: int = None,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        y_pred, y_true = self._ignore_flatten(y_pred, y_true)
        tp, tn, fp, fn = self._tfpn(y_pred, y_true)

        tversky_class = (tp + self.smooth) / (
            (tp + self.delta * fn + (1 - self.delta) * fp) + self.smooth
        )
        # Average class scores
        return torch.mean(torch.pow((1 - tversky_class), self.gamma))


################################
#          Focal loss          #
################################
class FocalLoss(_Loss):
    """Focal loss is used to address the issue of the class imbalance problem.
        A modulation term applied to the Cross-Entropy loss function.

    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives. alpha > 0.5
        penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples,
        by default 2.
    """

    def __init__(
        self, alpha: float = None, gamma_f: float = 2.0, ignore_index: int = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma_f = gamma_f
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        y_pred, y_true = self._ignore_flatten(y_pred, y_true)
        cross_entropy = -y_true * torch.log(y_pred + EPSILON)

        if self.alpha is not None:
            alpha_weight = torch.Tensor(self.alpha, dtype=torch.float32)
            focal_loss = (
                alpha_weight
                * torch.pow(1 - y_pred + EPSILON, self.gamma_f)
                * cross_entropy
            )
        else:
            focal_loss = torch.pow(1 - y_pred + EPSILON, self.gamma_f) * cross_entropy

        return torch.mean(torch.sum(focal_loss, dim=1))


################################
#          Combo loss          #
################################
class ComboLoss(_Loss):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798

    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss., by default 0.5
    beta : float, optional
        beta > 0.5 penalises false negatives more than false positives., by default 0.5
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, ignore_index: int = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.dice = DiceCoefficient(ignore_index=self.ignore_index)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        dice = self.dice(y_pred, y_true)
        y_pred, y_true = self._ignore_flatten(y_pred, y_true)
        cross_entropy = -y_true * torch.log(y_pred + EPSILON)

        if self.beta is not None:
            cross_entropy = self.beta * cross_entropy + (1 - self.beta) * cross_entropy

        # sum over classes
        cross_entropy = torch.mean(torch.sum(cross_entropy, dim=1))
        if self.alpha is not None:
            return (self.alpha * cross_entropy) - ((1 - self.alpha) * dice)
        else:
            return cross_entropy - dice


#################################
# Symmetric Focal Tversky loss  #
#################################
class SymmetricFocalTverskyLoss(_Loss):
    """This is the implementation for binary segmentation.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default
        0.75
    """

    def __init__(
        self, delta: float = 0.7, gamma: float = 0.75, ignore_index: int = None
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        y_pred, y_true = self._ignore_flatten(y_pred, y_true)
        tp, tn, fp, fn = self._tfpn(y_pred, y_true, reduction="none")

        # Calculate Dice score
        dice_class = (tp + EPSILON) / (
            (tp + self.delta * fn + (1 - self.delta) * fp) + EPSILON
        )

        # This assumes that the background class is the first class
        back_dice = (1 - dice_class[:, 0].unsqueeze(1)) * torch.pow(
            (1 - dice_class[:, 0].unsqueeze(1) + EPSILON), -self.gamma
        )
        fore_dice = (1 - dice_class[:, 1:]) * torch.pow(
            (1 - dice_class[:, 1:] + EPSILON), -self.gamma
        )

        # Average class scores
        return torch.mean(torch.concat([back_dice, fore_dice], dim=1))


#################################
# Asymmetric Focal Tversky loss #
#################################
class AsymmetricFocalTverskyLoss(_Loss):
    """This is the implementation for binary segmentation.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples,
        by default 0.75
    """

    def __init__(
        self, delta: float = 0.7, gamma: float = 0.75, ignore_index: int = None
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        y_pred, y_true = self._ignore_flatten(y_pred, y_true)
        tp, tn, fp, fn = self._tfpn(y_pred, y_true, reduction="none")

        # Calculate Dice score
        dice_class = (tp + EPSILON) / (
            (tp + self.delta * fn + (1 - self.delta) * fp) + EPSILON
        )

        # This assumes that the background class is the first class
        back_dice = 1 - dice_class[:, 0].unsqueeze(1)
        fore_dice = (1 - dice_class[:, 1:]) * torch.pow(
            (1 - dice_class[:, 1:] + EPSILON), -self.gamma
        )

        # Average class scores
        return torch.mean(torch.concat([back_dice, fore_dice], dim=1))


################################
#    Symmetric Focal loss      #
################################
class SymmetricFocalLoss(_Loss):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy
        examples, by default 2.0
    """

    def __init__(
        self, delta: float = 0.7, gamma: float = 2.0, ignore_index: int = None
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        y_pred, y_true = self._ignore_flatten(y_pred, y_true)
        cross_entropy = -y_true * torch.log(y_pred + EPSILON)

        back_ce = torch.pow(1 - y_pred[:, 0].unsqueeze(1), self.gamma) * cross_entropy[
            :, 0
        ].unsqueeze(1)
        back_ce = (1 - self.delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:, 1:], self.gamma) * cross_entropy[:, 1:]
        fore_ce = self.delta * fore_ce

        return torch.mean(torch.sum(torch.concat([back_ce, fore_ce], dim=1), dim=1))


################################
#     Asymmetric Focal loss    #
################################
class AsymmetricFocalLoss(_Loss):
    """For Imbalanced datasets

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of
        easy examples, by default 2.0
    """

    def __init__(
        self, delta: float = 0.7, gamma: float = 2.0, ignore_index: int = None
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        y_pred, y_true = self._ignore_flatten(y_pred, y_true)
        cross_entropy = -y_true * torch.log(y_pred + EPSILON)

        back_ce = torch.pow(1 - y_pred[:, 0].unsqueeze(1), self.gamma) * cross_entropy[
            :, 0
        ].unsqueeze(1)
        back_ce = (1 - self.delta) * back_ce

        fore_ce = self.delta * cross_entropy[:, 1:]

        return torch.mean(torch.sum(torch.concat([back_ce, fore_ce], dim=1), dim=1))


###########################################
#      Symmetric Unified Focal loss       #
###########################################
class SymUnifiedFocalLoss(_Loss):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based
        and cross entropy-based loss functions into a single framework.

    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal
        Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground
        enhancement, by default 0.5
    """

    def __init__(
        self,
        weight: float = 0.5,
        delta: float = 0.6,
        gamma: float = 0.5,
        ignore_index: int = None,
    ):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.ignore_index = ignore_index

        self.symmetric_ftl = SymmetricFocalTverskyLoss(
            delta=self.delta, gamma=self.gamma, ignore_index=self.ignore_index
        )
        self.symmetric_fl = SymmetricFocalLoss(
            delta=self.delta, gamma=self.gamma, ignore_index=self.ignore_index
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        symmetric_ftl = self.symmetric_ftl(y_pred, y_true)
        symmetric_fl = self.symmetric_fl(y_pred, y_true)
        if self.weight is not None:
            return (self.weight * symmetric_ftl) + ((1 - self.weight) * symmetric_fl)
        else:
            return symmetric_ftl + symmetric_fl


###########################################
#      Asymmetric Unified Focal loss      #
###########################################
class AsymUnifiedFocalLoss(_Loss):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based
        and cross entropy-based loss functions into a single framework.

    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal
        Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground
        enhancement, by default 0.5
    """

    def __init__(
        self,
        weight: float = 0.5,
        delta: float = 0.6,
        gamma: float = 0.5,
        ignore_index: int = None,
    ):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.ignore_index = ignore_index

        self.asymmetric_ftl = AsymmetricFocalTverskyLoss(
            delta=self.delta, gamma=self.gamma, ignore_index=self.ignore_index
        )
        self.asymmetric_fl = AsymmetricFocalLoss(
            delta=self.delta, gamma=self.gamma, ignore_index=self.ignore_index
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_pred : Tensor of shape (batch_size, num_classes, ...)
            Predicted labels as model softmax outputs.
        y_true : One-hot encoded tensor of shape (batch_size, num_classes, ...)
            Ground truth labels.

        Returns
        -------
        loss : Tensor of shape (1,)
            Loss value.
        """

        asymmetric_ftl = self.asymmetric_ftl(y_pred, y_true)
        asymmetric_fl = self.asymmetric_fl(y_pred, y_true)
        if self.weight is not None:
            return (self.weight * asymmetric_ftl) + ((1 - self.weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl
