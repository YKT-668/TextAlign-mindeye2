"""Reference implementations of the losses used by the Final14 trainer."""

import torch
import torch.nn.functional as F


def contrastive_alignment_loss(
    prediction: torch.Tensor, positive: torch.Tensor, tau: float = 0.07
) -> torch.Tensor:
    """Batchwise InfoNCE between predicted and positive text embeddings."""
    prediction = F.normalize(prediction.float(), dim=-1)
    positive = F.normalize(positive.float(), dim=-1)
    logits = prediction @ positive.T / tau
    labels = torch.arange(prediction.shape[0], device=prediction.device)
    return F.cross_entropy(logits, labels)


def counterfactual_margin_loss(
    prediction: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Require positive similarity to exceed counterfactual similarity."""
    prediction = F.normalize(prediction.float(), dim=-1)
    positive = F.normalize(positive.float(), dim=-1)
    negative = F.normalize(negative.float(), dim=-1)
    pos_similarity = (prediction * positive).sum(dim=-1)
    neg_similarity = (prediction * negative).sum(dim=-1)
    return F.relu(margin - pos_similarity + neg_similarity).mean()
