import torch
import torch.nn as nn
import torch.nn.functional as F


def sup_contrastive_loss(inputs, labels, tau=0.1):
    """
    better
    :param inputs: torch.Tensor, shape [batch_size, projection_dim]
    :param labels: torch.Tensor, shape [batch_size]
    :return: torch.Tensor, scalar
    """
    device = torch.device("cuda") if inputs.is_cuda else torch.device("cpu")

    inputs = F.normalize(inputs, dim=-1)

    dot_product_tempered = torch.mm(inputs, inputs.T) / tau
    # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
    exp_dot_tempered = (
        torch.exp(dot_product_tempered -
                  torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
    )

    mask_similar_class = (labels.unsqueeze(1).repeat(
        1, labels.shape[0]) == labels).to(device)
    mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
    mask_combined = mask_similar_class * mask_anchor_out
    cardinality_per_samples = torch.sum(mask_combined, dim=1) + 1e-8
    # print(cardinality_per_samples)

    log_prob = -torch.log(exp_dot_tempered /
                          (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
    supervised_contrastive_loss_per_sample = torch.sum(
        log_prob * mask_combined, dim=1) / cardinality_per_samples
    supervised_contrastive_loss = torch.mean(
        supervised_contrastive_loss_per_sample)
    return supervised_contrastive_loss
