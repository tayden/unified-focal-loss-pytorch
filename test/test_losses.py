import torch
import torch.nn.functional as F
from einops import repeat

import ufl as losses


def _to_batch_one_hot(t, num_classes=2):
    one_hot = F.one_hot(torch.LongTensor(t), num_classes=num_classes)
    return repeat(one_hot, "h w c -> n c h w", n=1).to(torch.float32)


TARGET_IMG = _to_batch_one_hot([[0, 0], [1, 1]])
TARGET_IMG_W_IGNORE = _to_batch_one_hot([[0, 0], [1, 1], [2, 2]], num_classes=3)
TARGET_IMG_FOUR_CLS = _to_batch_one_hot([[0, 0], [1, 1]], num_classes=4)


def test_dice_loss():
    loss = losses.DiceLoss()

    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = F.softmax(_to_batch_one_hot([[1, 1], [0, 0]]) + 0.2, dim=1)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[0, 1], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = F.softmax(_to_batch_one_hot([[0, 1], [0, 1]]) + 0.4, dim=1)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.25

    preds = _to_batch_one_hot([[0, 1], [1, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.25

    preds = _to_batch_one_hot([[0, 1], [1, 1]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 0.25

    loss = losses.DiceLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [1, 1], [1, 1]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.25


def test_dice_coefficient():
    loss = losses.DiceCoefficient()

    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.0

    preds = F.softmax(_to_batch_one_hot([[1, 1], [0, 0]]) + 0.2, dim=1)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[0, 1], [1, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = F.softmax(_to_batch_one_hot([[0, 1], [1, 0]]) + 0.4, dim=1)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = _to_batch_one_hot([[0, 0], [1, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.75

    preds = _to_batch_one_hot([[0, 1], [1, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.75

    preds = _to_batch_one_hot([[0, 1], [1, 1]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 0.75

    loss = losses.DiceCoefficient(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [1, 1], [1, 1]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.75


def test_tversky_loss():
    loss = losses.TverskyLoss()
    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.25

    preds = _to_batch_one_hot([[0, 0], [0, 1]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 0.25

    loss = losses.TverskyLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 0], [0, 1], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.25


def test_focal_tversky_loss():
    loss = losses.FocalTverskyLoss()
    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.595

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.354

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.806

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 0.806

    loss = losses.FocalTverskyLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.806


def test_combo_loss():
    loss = losses.ComboLoss()
    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == -0.5

    preds = torch.Tensor([[[[0.9, 0.9], [0.1, 0.1]], [[0.1, 0.1], [0.9, 0.9]]]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == -0.447

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 11.513

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 5.506

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 2.503

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 8.51

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 8.51

    preds = torch.Tensor(
        [
            [
                [[0.9, 0.1], [0.9, 0.9]],
                [[0.1, 0.9], [0.1, 0.1]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 0.752

    loss = losses.ComboLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 8.51


def test_focal_loss():
    loss = losses.FocalLoss()
    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 23.026

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 11.513

    preds = _to_batch_one_hot([[0, 0], [0, 0]]) + 0.2
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.511

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 5.756

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 17.269

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 17.269

    loss = losses.FocalLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 17.269


def test_sym_focal_loss():
    loss = losses.SymmetricFocalLoss()
    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 11.513

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 8.059

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 4.03

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 9.786

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 9.786

    loss = losses.SymmetricFocalLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 9.786


def test_sym_focal_tversky_loss():
    loss = losses.SymmetricFocalTverskyLoss()
    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.25

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.75

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 0.375

    loss = losses.SymmetricFocalTverskyLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.75


def test_asym_focal_loss():
    loss = losses.AsymmetricFocalLoss()
    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 11.513

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 8.059

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 4.03

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 9.786

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 9.786

    loss = losses.AsymmetricFocalLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 9.786


def test_asym_focal_tversky_loss():
    loss = losses.AsymmetricFocalTverskyLoss()
    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.25

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.75

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 0.375

    loss = losses.AsymmetricFocalTverskyLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.75


def test_sym_unified_focal_loss():
    loss = losses.SymUnifiedFocalLoss()
    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 6.256

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 3.704

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.852

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 4.98

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 4.793

    loss = losses.SymUnifiedFocalLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 4.98


def test_asym_unified_focal_loss():
    loss = losses.AsymUnifiedFocalLoss()
    assert round(float(loss(TARGET_IMG, TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 6.256

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 3.704

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.852

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 4.98

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG_FOUR_CLS)), 3) == 4.793

    loss = losses.AsymUnifiedFocalLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 4.98
