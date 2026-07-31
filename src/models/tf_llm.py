"""
GPT-2 Integration for TF-LLM.

Wires together the TD encoder and FD encoder outputs into a frozen GPT-2
backbone, per Section 3.3 and Eq. 7 of the paper:

    Z_pred = concat(T_Pro, T_Seq, dup(T_Mask, l_f))

Where:
    T_Pro  = learnable prompt tokens (Section 3.3: "prompt tokens ... are
             learnable embeddings ... trained to ensure they capture the
             implicit context within the time series")
    T_Seq  = sequence tokens, from the fused time-frequency patch embeddings
    T_Mask = mask tokens, added in Stage 7 for the forecasting task head

This file implements T_Pro and T_Seq, and the frozen GPT-2 backbone that
consumes them. T_Mask concatenation and forecasting-specific logic is
implemented in Stage 7's task head, per Eq. 7.

IMPLEMENTATION NOTE (documented deviation, due to paper ambiguity):
The TD encoder produces per-patch embeddings (num_patches, d_emb) while the
FD encoder produces a pooled global embedding (d_emb,). The paper describes
a "WeightedSum" fusion strategy but does not specify how to reconcile these
different sequence lengths. We fuse by broadcasting the pooled frequency
embedding as global context into every time patch:
    fused_patch_i = alpha * td_patch_i + (1 - alpha) * z_F
`alpha` is a learnable scalar, initialized at 0.5 (neutral). This fusion
happens upstream of Eq. 7's T_Seq, consistent with Fig. 2's depiction of
fusion occurring before the LLM."""


import torch
import torch.nn as nn
from transformers import GPT2Model


class TFLLMBackbone(nn.Module):
    """
    Combines fused TD/FD sequence embeddings (T_Seq) with learnable prompt
    tokens (T_Pro), feeds the result through a frozen GPT-2, and returns
    the output hidden states for downstream task heads (which append
    T_Mask, per Eq. 7).
    """

    def __init__(self, d_emb=64, num_prompt_tokens=10, gpt2_name="gpt2", dropout=0.1):
        super().__init__()

        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)
        self.hidden_size = self.gpt2.config.hidden_size  # 768 for gpt2-small

        # freeze the entire backbone
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # --- trainable components ---

        # fusion weight for combining TD patch embeddings with pooled FD embedding
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

        # project fused d_emb-dim sequence tokens into GPT-2's hidden size
        self.input_proj = nn.Sequential(
            nn.Linear(d_emb, self.hidden_size),
            nn.Dropout(dropout),
        )

        # T_Pro: learnable prompt tokens (Section 3.3)
        self.prompt_tokens = nn.Parameter(
            torch.randn(num_prompt_tokens, self.hidden_size) * 0.02
        )
        self.num_prompt_tokens = num_prompt_tokens

    def fuse_td_fd(self, td_patch_embeddings, fd_pooled_embedding):
        """
        Args:
            td_patch_embeddings: (batch, num_patches, d_emb)
            fd_pooled_embedding: (batch, d_emb)
        Returns:
            fused: (batch, num_patches, d_emb)
        """
        fd_broadcast = fd_pooled_embedding.unsqueeze(1).expand_as(td_patch_embeddings)
        alpha = torch.sigmoid(self.fusion_alpha)  # keep in [0, 1]
        fused = alpha * td_patch_embeddings + (1 - alpha) * fd_broadcast
        return fused

    def build_sequence_tokens(self, td_patch_embeddings, fd_pooled_embedding):
        """
        Produces T_Seq: fused, projected sequence tokens.

        Args:
            td_patch_embeddings: (batch, num_patches, d_emb)
            fd_pooled_embedding: (batch, d_emb)
        Returns:
            seq_tokens: (batch, num_patches, hidden_size)
        """
        fused = self.fuse_td_fd(td_patch_embeddings, fd_pooled_embedding)
        seq_tokens = self.input_proj(fused)
        return seq_tokens

    def get_prompt_tokens(self, batch_size):
        """Returns T_Pro, expanded to the batch size."""
        return self.prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def forward_backbone(self, input_embeds, attention_mask=None):
        """
        Feeds an already-assembled input sequence (T_Pro + T_Seq + T_Mask,
        or any subset thereof) through the frozen GPT-2 backbone.

        Args:
            input_embeds: (batch, total_seq_len, hidden_size)
            attention_mask: (batch, total_seq_len), optional
        Returns:
            hidden_states: (batch, total_seq_len, hidden_size)
        """
        outputs = self.gpt2(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def forward(self, td_patch_embeddings, fd_pooled_embedding):
        """
        Base forward pass implementing Eq. 7's T_Pro + T_Seq concatenation
        (without T_Mask — that is task-specific and added in Stage 7).

        Args:
            td_patch_embeddings: (batch, num_patches, d_emb)
            fd_pooled_embedding: (batch, d_emb)
        Returns:
            hidden_states: (batch, num_prompt_tokens + num_patches, hidden_size)
        """
        batch_size = td_patch_embeddings.shape[0]

        seq_tokens = self.build_sequence_tokens(td_patch_embeddings, fd_pooled_embedding)
        prompt_tokens = self.get_prompt_tokens(batch_size)

        full_input = torch.cat([prompt_tokens, seq_tokens], dim=1)  # T_Pro + T_Seq

        hidden_states = self.forward_backbone(full_input)
        return hidden_states

    def count_parameters(self):
        """Returns (trainable_params, total_params) for sanity checking."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


if __name__ == "__main__":
    # quick smoke test
    import sys
    sys.path.append(".")
    from td_encoder import TDEncoder
    from fd_encoder import FDEncoder

    batch_size = 2
    seq_len = 512
    patch_len = 16
    num_features = 7
    d_emb = 64

    x = torch.randn(batch_size, seq_len, num_features)
    x_patched = x.reshape(batch_size, seq_len // patch_len, patch_len, num_features)

    td_encoder = TDEncoder(patch_len=patch_len, num_features=num_features, d_emb=d_emb)
    fd_encoder = FDEncoder(seq_len=seq_len, num_features=num_features, d_emb=d_emb)

    _, td_patch_embeddings = td_encoder(x_patched)   # (B, num_patches, d_emb)
    z_F, _ = fd_encoder(x)                             # (B, d_emb)

    backbone = TFLLMBackbone(d_emb=d_emb, num_prompt_tokens=10)

    trainable, total = backbone.count_parameters()
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Frozen fraction:  {100 * (1 - trainable / total):.2f}%")
    print()

    hidden_states = backbone(td_patch_embeddings, z_F)
    print(f"Output hidden_states shape: {tuple(hidden_states.shape)}")
    print(f"Expected seq len: num_prompt_tokens(10) + num_patches(32) = 42")
