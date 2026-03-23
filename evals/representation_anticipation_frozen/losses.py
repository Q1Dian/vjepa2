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


def topk_representation_loss(predictions, targets, topk_ratio=0.25, reduction="mean"):
    """Compute a top-k cosine loss over token-wise latent predictions.

    predictions and targets are expected to have shape [B, N, D]. The loss is
    the mean of the largest token-wise cosine distances per sample.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"predictions and targets must match, got {predictions.shape} vs {targets.shape}")

    token_loss = 1.0 - F.cosine_similarity(predictions, targets, dim=-1)
    token_count = token_loss.size(1)
    if token_count == 0:
        raise ValueError("topk_representation_loss received zero tokens")

    safe_ratio = min(1.0, max(0.0, float(topk_ratio)))
    topk = max(1, int(token_count * safe_ratio))
    topk_loss = token_loss.topk(topk, dim=1).values.mean(dim=1)

    if reduction == "mean":
        return topk_loss.mean()
    if reduction == "sum":
        return topk_loss.sum()
    return topk_loss
