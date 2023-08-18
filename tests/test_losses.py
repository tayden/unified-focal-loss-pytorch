import torch
import torch.nn.functional as F
from einops import repeat

import unified_focal_loss as losses


def _to_batch_one_hot(t, num_classes=2):
    one_hot = F.one_hot(torch.LongTensor(t), num_classes=num_classes)
    return repeat(one_hot, "h w c -> n c h w", n=1).to(torch.float32)


TARGET_IMG = torch.LongTensor([[0, 0], [1, 1]])
TARGET_IMG_W_IGNORE = torch.LongTensor([[0, 0], [1, 1], [2, 2]])


def test_dice_loss():
    loss = losses.DiceLoss()

    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = F.softmax(_to_batch_one_hot([[1, 1], [0, 0]]) + 0.2, dim=1)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.731

    preds = _to_batch_one_hot([[0, 1], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = F.softmax(_to_batch_one_hot([[0, 1], [0, 1]]) + 0.4, dim=1)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.267

    preds = _to_batch_one_hot([[0, 1], [1, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.267

    preds = _to_batch_one_hot([[0, 1], [1, 1]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.133

    loss = losses.DiceLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [1, 1], [1, 1]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.267


def test_dice_coefficient():
    loss = losses.DiceCoefficient()

    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.0

    preds = F.softmax(_to_batch_one_hot([[1, 1], [0, 0]]) + 0.2, dim=1)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.269

    preds = _to_batch_one_hot([[0, 1], [1, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = F.softmax(_to_batch_one_hot([[0, 1], [1, 0]]) + 0.4, dim=1)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = _to_batch_one_hot([[0, 0], [1, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.733

    preds = _to_batch_one_hot([[0, 1], [1, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.733

    preds = _to_batch_one_hot([[0, 1], [1, 1]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.867

    loss = losses.DiceCoefficient(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [1, 1], [1, 1]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.733


def test_tversky_loss():
    loss = losses.TverskyLoss()
    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.615

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.271

    preds = _to_batch_one_hot([[0, 0], [0, 1]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.136

    loss = losses.TverskyLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 0], [0, 1], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.271


def test_focal_tversky_loss():
    loss = losses.FocalTverskyLoss()
    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.666

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.366

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.826

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.413

    loss = losses.FocalTverskyLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.826


def test_combo_loss():
    loss = losses.ComboLoss()
    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == -0.5

    preds = torch.Tensor([[[[0.9, 0.9], [0.1, 0.1]], [[0.1, 0.1], [0.9, 0.9]]]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == -0.397

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 9.21

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 4.439

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.936

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 6.808

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 6.608

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
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.559

    loss = losses.ComboLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 6.808


def test_focal_loss():
    loss = losses.FocalLoss()
    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 12.894

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 6.447

    preds = F.softmax(_to_batch_one_hot([[0, 0], [0, 0]]) + 0.2)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.404

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 3.224

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 9.671

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 9.671

    loss = losses.FocalLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 9.671


def test_sym_focal_loss():
    loss = losses.SymmetricFocalLoss()
    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 9.21

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 6.447

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 3.224

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 7.829

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 7.829

    loss = losses.SymmetricFocalLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 7.829


def test_sym_focal_tversky_loss():
    loss = losses.SymmetricFocalTverskyLoss()
    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.25

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.75

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.375

    loss = losses.SymmetricFocalTverskyLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.75


def test_asym_focal_loss():
    loss = losses.AsymmetricFocalLoss()
    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 9.21

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 6.447

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 3.224

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 7.829

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 7.829

    loss = losses.AsymmetricFocalLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 7.829


def test_asym_focal_tversky_loss():
    loss = losses.AsymmetricFocalTverskyLoss()
    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.0

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.5

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.25

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.75

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 0.375

    loss = losses.AsymmetricFocalTverskyLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 0.75


def test_sym_unified_focal_loss():
    loss = losses.SymmetricUnifiedFocalLoss()
    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 5.105

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 3.474

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.737

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 4.289

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 4.102

    loss = losses.SymmetricUnifiedFocalLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 4.289


def test_asym_unified_focal_loss():
    loss = losses.AsymmetricUnifiedFocalLoss()
    assert round(float(loss(_to_batch_one_hot(TARGET_IMG, 2), TARGET_IMG)), 3) == 0.0

    preds = _to_batch_one_hot([[1, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 5.105

    preds = _to_batch_one_hot([[0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 3.474

    preds = _to_batch_one_hot([[0, 0], [0, 1]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 1.737

    preds = _to_batch_one_hot([[0, 1], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG)), 3) == 4.289

    preds = _to_batch_one_hot([[0, 1], [0, 0]], num_classes=4)
    assert round(float(loss(preds, TARGET_IMG)), 3) == 4.102

    loss = losses.AsymmetricUnifiedFocalLoss(ignore_index=2)
    preds = _to_batch_one_hot([[0, 1], [0, 0], [0, 0]])
    assert round(float(loss(preds, TARGET_IMG_W_IGNORE)), 3) == 4.289
