# Unified Focal Loss PyTorch

An implementation of loss functions
from [“Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation”][1]

Extended for multiclass classification and to allow passing an ignore index.

*Note: This implementation is not tested against the original implementation. It varies
from the original implementation based on my own interpretation of the paper.*

[1]: https://github.com/mlyg/unified-focal-loss

## Installation

```bash
pip install unified-focal-loss-pytorch
```

## Usage

```python
import torch
import torch.nn.functional as F
from unified_focal_loss import AsymmetricUnifiedFocalLoss

loss_fn = AsymmetricUnifiedFocalLoss(
    delta=0.7,
    gamma=0.5,
    ignore_index=2,
)

logits = torch.tensor([
    [[0.1000, 0.4000],
     [0.2000, 0.5000],
     [0.3000, 0.6000]],

    [[0.7000, 0.0000],
     [0.8000, 0.1000],
     [0.9000, 0.2000]]
])

# Shape should be (batch_size, num_classes, ...)
probs = F.softmax(logits, dim=1)
# Shape should be (batch_size, ...). Not one-hot encoded.
targets = torch.tensor([
    [0, 1],
    [2, 0],
])

loss = loss_fn(probs, targets)
print(loss)
# >>> tensor(0.6737)
```
