<!-- markdownlint-disable -->

<a href="../unified_focal_loss/losses.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `unified_focal_loss.losses`






---

<a href="../unified_focal_loss/losses.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DiceCoefficient`
The Dice similarity coefficient, also known as the Sørensen–Dice index or simply  Dice coefficient, is a statistical tool which measures the similarity between  two sets of data.



**Args:**

 - <b>`delta `</b>:  float, optional  controls weight given to false positive and false negatives, by default 0.7.
 - <b>`smooth `</b>:  float, optional  smoothing constant to prevent division by zero errors, by default 0.000001.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    delta: 'float' = 0.7,
    smooth: 'float' = 1e-08,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Tensor of shape (batch_size, ...)  Ground truth labels.



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DiceLoss`
Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in the 1940s to gauge the similarity between two samples.



**Args:**

 - <b>`delta `</b>:  float, optional  controls weight given to false positive and false negatives, by default 0.7.
 - <b>`smooth `</b>:  float, optional  smoothing constant to prevent division by zero errors, by default 0.000001.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    delta: 'float' = 0.7,
    smooth: 'float' = 1e-08,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Tensor of shape (batch_size, ...)  Ground truth labels.



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TverskyLoss`
Tversky loss function for image segmentation using 3D fully convolutional deep networks. Link: https://arxiv.org/abs/1706.05721



**Args:**

 - <b>`delta `</b>:  float, optional  controls weight given to false positive and false negatives, by default 0.7.
 - <b>`smooth `</b>:  float, optional  smoothing constant to prevent division by zero errors, by default 0.000001.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    delta: 'float' = 0.7,
    smooth: 'float' = 1e-08,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Tensor of shape (batch_size, ...)  Ground truth labels.



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FocalTverskyLoss`
A Novel Focal Tversky loss function with improved Attention U-Net for lesion  segmentation Link: https://arxiv.org/abs/1810.07842



**Args:**

 - <b>`delta `</b>:  float, optional  controls weight given to each class, by default 0.7
 - <b>`gamma `</b>:  float, optional  focal parameter controls degree of down-weighting of easy examples,  by default 0.75
 - <b>`smooth `</b>:  float, optional  smoothing constant to prevent division by zero errors, by default 0.000001.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    delta: 'float' = 0.7,
    gamma: 'float' = 0.75,
    smooth: 'float' = 1e-08,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Tensor of shape (batch_size, ...)  Ground truth labels.



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L266"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FocalLoss`
Focal loss is used to address the issue of the class imbalance problem.  A modulation term applied to the Cross-Entropy loss function.



**Args:**

 - <b>`delta `</b>:  float, optional  controls relative weight of false positives and false negatives. delta > 0.5  penalises false negatives more than false positives, by default 0.7.
 - <b>`gamma `</b>:  float, optional  focal parameter controls degree of down-weighting of easy examples,  by default 0.75.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L281"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    delta: 'float' = 0.7,
    gamma: 'float' = 0.75,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Tensor of shape (batch_size, ...)  Ground truth labels.



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ComboLoss`
Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation Link: https://arxiv.org/abs/1805.02798



**Args:**

 - <b>`alpha `</b>:  float, optional  controls weighting of dice and cross-entropy loss., by default 0.5.
 - <b>`beta `</b>:  float, optional  beta > 0.5 penalises false negatives more than false positives., by default 0.5.
 - <b>`smooth `</b>:  float, optional  smoothing constant to prevent division by zero errors, by default 0.000001.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    alpha: 'float' = 0.5,
    beta: 'float' = 0.5,
    smooth: 'float' = 1e-08,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Tensor of shape (batch_size, ...)  Ground truth labels.



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L354"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SymmetricFocalTverskyLoss`
This is the implementation for binary segmentation.



**Args:**

 - <b>`delta `</b>:  float, optional  controls weight given to false positive and false negatives, by default 0.7.
 - <b>`gamma `</b>:  float, optional  focal parameter controls degree of down-weighting of easy examples, by default  0.75.
 - <b>`smooth `</b>:  float, optional  smoothing constant to prevent division by zero errors, by default 0.000001.
 - <b>`common_class_index `</b>:  int, optional  index of the common class, by default 0.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L372"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    delta: 'float' = 0.7,
    gamma: 'float' = 0.75,
    smooth: 'float' = 1e-08,
    common_class_index: 'int' = 0,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Tensor of shape (batch_size, ...)  Ground truth labels.



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L420"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AsymmetricFocalTverskyLoss`
This is the implementation for binary segmentation.



**Args:**

 - <b>`delta `</b>:  float, optional  controls weight given to false positive and false negatives, by default 0.7
 - <b>`gamma `</b>:  float, optional  focal parameter controls degree of down-weighting of easy examples,  by default 0.75.
 - <b>`smooth `</b>:  float, optional  smoothing constant to prevent division by zero errors, by default 0.000001.
 - <b>`common_class_index `</b>:  int, optional  index of the common class, by default 0.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    delta: 'float' = 0.7,
    gamma: 'float' = 0.75,
    smooth: 'float' = 1e-08,
    common_class_index: 'int' = 0,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Tensor of shape (batch_size, ...)  Ground truth labels.



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L480"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SymmetricFocalLoss`


**Args:**

 - <b>`delta `</b>:  float, optional  controls weight given to false positive and false negatives, by default 0.7.
 - <b>`gamma `</b>:  float, optional  Focal Tversky loss' focal parameter controls degree of down-weighting of  easy examples, by default 0.75.
 - <b>`smooth `</b>:  float, optional  smoothing constant to prevent division by zero errors, by default 0.000001.
 - <b>`common_class_index `</b>:  int, optional  index of the common class, by default 0.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L496"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    delta: 'float' = 0.7,
    gamma: 'float' = 0.75,
    smooth: 'float' = 1e-08,
    common_class_index: 'int' = 0,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Tensor of shape (batch_size, ...)  Ground truth labels.



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L535"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AsymmetricFocalLoss`
For Imbalanced datasets



**Args:**

 - <b>`delta `</b>:  float, optional  controls weight given to false positive and false negatives, by default 0.7.
 - <b>`gamma `</b>:  float, optional  Focal Tversky loss' focal parameter controls degree of down-weighting of  easy examples, by default 0.75.
 - <b>`common_class_index `</b>:  int, optional  index of the common class, by default 0.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L550"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    delta: 'float' = 0.7,
    gamma: 'float' = 0.75,
    common_class_index: 'int' = 0,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Tensor of shape (batch_size, ...)  Ground truth labels.



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L585"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SymmetricUnifiedFocalLoss`
The Unified Focal loss is a new compound loss function that unifies Dice-based  and cross entropy-based loss functions into a single framework.



**Args:**

 - <b>`weight `</b>:  float, optional  represents lambda parameter and controls weight given to symmetric Focal  Tversky loss and symmetric Focal loss, by default 0.5.
 - <b>`delta `</b>:  float, optional  controls weight given to each class, by default 0.7.
 - <b>`gamma `</b>:  float, optional  focal parameter controls the degree of background suppression and foreground  enhancement, by default 0.75.
 - <b>`common_class_index `</b>:  int, optional  index of the common class, by default 0.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L604"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    weight: 'float' = 0.5,
    delta: 'float' = 0.7,
    gamma: 'float' = 0.75,
    common_class_index: 'int' = 0,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L628"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Ground truth labels Tensor of shape (batch_size, ...).



**Returns:**

 - <b>`loss `</b>:  Loss value.


---

<a href="../unified_focal_loss/losses.py#L651"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AsymmetricUnifiedFocalLoss`
The Unified Focal loss is a new compound loss function that unifies Dice-based  and cross entropy-based loss functions into a single framework.



**Args:**

 - <b>`weight `</b>:  float, optional  represents lambda parameter and controls weight given to asymmetric Focal  Tversky loss and asymmetric Focal loss, by default 0.5.
 - <b>`delta `</b>:  float, optional  controls weight given to each class, by default 0.7
 - <b>`gamma `</b>:  float, optional  focal parameter controls the degree of background suppression and foreground  enhancement, by default 0.75.
 - <b>`common_class_index `</b>:  int, optional  index of the common class, by default 0.
 - <b>`ignore_index `</b>:  int, optional  index of the ignore class, by default None.

<a href="../unified_focal_loss/losses.py#L670"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    weight: 'float' = 0.5,
    delta: 'float' = 0.7,
    gamma: 'float' = 0.75,
    common_class_index: 'int' = 0,
    ignore_index: 'int' = None
)
```








---

<a href="../unified_focal_loss/losses.py#L694"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(y_pred: 'Tensor', y_true: 'Tensor') → Tensor
```

Calculate loss.



**Args:**

 - <b>`y_pred `</b>:  Tensor of shape (batch_size, num_classes, ...).  Predicted probabilities for each output class.
 - <b>`y_true `</b>:  Ground truth labels Tensor of shape (batch_size, ...).



**Returns:**

 - <b>`loss `</b>:  Loss value.
