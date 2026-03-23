# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs,
    targets,
    alpha=0.25,
    gamma=2.0,
    reduction="sum",
    detach=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    :param Tensor inputs: Prediction logits for each sample [B x K]
    :param Tensor targets: Class label for each sample [B] (long tensor)
    :param float alpha: Weight in range (0,1) to balance pos vs neg samples.
    :param float gamma: Exponent of modulating factor (1-p_t) to balance easy vs hard samples.
    :param str reduction: 'mean' | 'sum'
    """
    B, K = inputs.size()  # [batch_size, class logits]

    # convert to one-hot targets
    targets = F.one_hot(targets, K).float()  # [B, K]

    p = F.sigmoid(inputs)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    if detach:
        loss = loss.detach()

    return loss


def topk_representation_loss(predictions, targets, topk_ratio=0.25, reduction="mean", patches_per_step=None):
    """Compute top-k patch L1 loss for each temporal step.

    predictions and targets are expected to have shape [B, N, D], where N is
    a flattened sequence of temporal steps and spatial patches. If
    ``patches_per_step`` is provided and divides N, top-k is applied across
    patches independently for each temporal step.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"predictions and targets must match, got {predictions.shape} vs {targets.shape}")

    patch_loss = F.l1_loss(predictions, targets, reduction="none").mean(dim=-1)
    token_count = patch_loss.size(1)
    if token_count == 0:
        raise ValueError("topk_representation_loss received zero tokens")

    safe_ratio = min(1.0, max(0.0, float(topk_ratio)))
    if patches_per_step is not None and patches_per_step > 0 and token_count % patches_per_step == 0:
        temporal_steps = token_count // patches_per_step
        patch_loss = patch_loss.reshape(patch_loss.size(0), temporal_steps, patches_per_step)
        topk = max(1, int(patches_per_step * safe_ratio))
        topk_loss = patch_loss.topk(topk, dim=2).values.mean(dim=(1, 2))
    else:
        topk = max(1, int(token_count * safe_ratio))
        topk_loss = patch_loss.topk(topk, dim=1).values.mean(dim=1)

    if reduction == "mean":
        return topk_loss.mean()
    if reduction == "sum":
        return topk_loss.sum()
    return topk_loss
