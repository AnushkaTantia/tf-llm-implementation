"""
NT-Xent (normalized temperature-scaled cross-entropy) contrastive loss.

Implements Eq. 1 / Eq. 2 from the TF-LLM paper:
    L = -log( exp(sim(z_i, z_i_aug) / tau) / sum_j exp(sim(z_i, z_j) / tau) )

Used identically for both the TD and FD contrastive encoders — same loss,
different input embeddings.
"""

import torch
import torch.nn.functional as F


def nt_xent_loss(z, z_aug, temperature=0.5):
    """
    Args:
        z:      (batch_size, d_emb) — original sample embeddings
        z_aug:  (batch_size, d_emb) — augmented sample embeddings (positive pairs with z)
        temperature: tau, scales the similarity before softmax

    Returns:
        scalar loss (averaged over the batch)

    For each sample i in the batch:
        - positive pair: (z_i, z_aug_i)
        - negative pairs: (z_i, z_j) and (z_i, z_aug_j) for all j != i
    """
    batch_size = z.shape[0]
    device = z.device

    # normalize embeddings so dot product == cosine similarity
    z = F.normalize(z, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)

    # concatenate: first half original, second half augmented
    # representations = [z_0, ..., z_{n-1}, z_aug_0, ..., z_aug_{n-1}]
    representations = torch.cat([z, z_aug], dim=0)  # (2*batch_size, d_emb)

    # full pairwise similarity matrix
    sim_matrix = torch.matmul(representations, representations.T) / temperature  # (2B, 2B)

    # mask out self-similarity (diagonal) — a sample is never its own negative/positive
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(self_mask, float("-inf"))

    # positive pair indices: for row i in [0, B) the positive is at i+B, and vice versa
    positive_idx = torch.arange(2 * batch_size, device=device)
    positive_idx = (positive_idx + batch_size) % (2 * batch_size)

    # cross-entropy where the "correct class" for row i is its positive pair index
    loss = F.cross_entropy(sim_matrix, positive_idx)

    return loss
