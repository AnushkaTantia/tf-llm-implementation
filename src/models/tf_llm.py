"""
GPT-2 Integration for TF-LLM — channel-independent version.

T_Seq carries a genuine variable axis: (batch, num_patches, num_variables,
d_emb), flattened into a single sequence dimension of length
(num_patches * num_variables) before feeding into GPT-2.

Includes post-fusion normalization per Fig. 2's caption: "the fused
features undergo embedding, normalization, and patching."

Eq. 7: Z_pred = concat(T_Pro, T_Seq, dup(T_Mask, l_f))
"""

import re

import torch
import torch.nn as nn
from transformers import GPT2Model


# GPT-2 parameters that the "norms_pos" policy trains (GPT4TS, Zhou et al., 2023):
# every block's two layer norms, the final layer norm, and the position embeddings.
# Attention, MLP and token-embedding weights always stay frozen.
LLM_NORMS_POS_PATTERN = re.compile(r"^(h\.\d+\.ln_[12]\.(weight|bias)|ln_f\.(weight|bias)|wpe\.weight)$")
LLM_POLICIES = ("frozen", "norms_pos")


class TFLLMBackbone(nn.Module):
    def __init__(self, d_emb=64, num_prompt_tokens=10, gpt2_name="gpt2", dropout=0.1, *,
                 use_llm, llm_policy, llm_hidden_size=768):
        """
        use_llm=True : load pretrained GPT-2 and apply `llm_policy`:
                       "frozen"    -> no GPT-2 parameter is trained
                       "norms_pos" -> only layer norms + position embeddings are trained
        use_llm=False: GPT-2 is not created at all (no-LLM ablation). Sequence tokens are
                       still projected to `llm_hidden_size` so that the forecasting head has
                       the same input size with and without the LLM.
        """
        super().__init__()
        self.use_llm = bool(use_llm)
        if self.use_llm:
            assert llm_policy in LLM_POLICIES, f"llm_policy must be one of {LLM_POLICIES}"
            self.gpt2 = GPT2Model.from_pretrained(gpt2_name)
            self.hidden_size = self.gpt2.config.hidden_size
            # 1) freeze ALL GPT-2 parameters explicitly ...
            for param in self.gpt2.parameters():
                param.requires_grad = False
            # 2) ... then unfreeze only what the policy allows
            if llm_policy == "norms_pos":
                for name, param in self.gpt2.named_parameters():
                    if LLM_NORMS_POS_PATTERN.match(name):
                        param.requires_grad = True
        else:
            assert llm_policy in (None, "none"), "llm_policy must be None/'none' when use_llm=False"
            self.gpt2 = None
            self.hidden_size = llm_hidden_size
        self.llm_policy = llm_policy if self.use_llm else "none"

        # fusion weight combining TD patch embeddings with per-variable pooled FD embeddings
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

        # normalization applied to the fused representation, per Fig. 2's caption
        self.fusion_norm = nn.LayerNorm(d_emb)

        # project fused d_emb-dim tokens into GPT-2's hidden size
        self.input_proj = nn.Sequential(
            nn.Linear(d_emb, self.hidden_size),
            nn.Dropout(dropout),
        )

        # T_Pro: learnable prompt tokens (Section 3.3). Only used with the LLM.
        self.prompt_tokens = nn.Parameter(
            torch.randn(num_prompt_tokens, self.hidden_size) * 0.02,
            requires_grad=self.use_llm,
        )
        self.num_prompt_tokens = num_prompt_tokens

    def fuse_td_fd(self, td_patch_embeddings, fd_pooled_per_variable):
        """
        Args:
            td_patch_embeddings: (batch, num_patches, num_variables, d_emb)
            fd_pooled_per_variable: (batch, num_variables, d_emb)
        Returns:
            fused: (batch, num_patches, num_variables, d_emb)
        """
        fd_broadcast = fd_pooled_per_variable.unsqueeze(1).expand_as(td_patch_embeddings)
        alpha = torch.sigmoid(self.fusion_alpha)
        fused = alpha * td_patch_embeddings + (1 - alpha) * fd_broadcast
        return fused

    def build_sequence_tokens(self, td_patch_embeddings, fd_pooled_per_variable):
        """
        Produces T_Seq, flattened to a single sequence axis for GPT-2.

        Returns:
            seq_tokens: (batch, num_patches * num_variables, hidden_size)
        """
        batch, num_patches, num_variables, d_emb = td_patch_embeddings.shape

        fused = self.fuse_td_fd(td_patch_embeddings, fd_pooled_per_variable)
        fused = self.fusion_norm(fused)
        fused_flat = fused.reshape(batch, num_patches * num_variables, d_emb)

        seq_tokens = self.input_proj(fused_flat)
        return seq_tokens

    def get_prompt_tokens(self, batch_size):
        return self.prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def forward_backbone(self, input_embeds, attention_mask=None):
        assert self.use_llm, "forward_backbone called with use_llm=False"
        outputs = self.gpt2(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def forward(self, td_patch_embeddings, fd_pooled_per_variable):
        """Base forward: T_Pro + T_Seq (T_Mask added by the task head)."""
        batch_size = td_patch_embeddings.shape[0]

        seq_tokens = self.build_sequence_tokens(td_patch_embeddings, fd_pooled_per_variable)
        prompt_tokens = self.get_prompt_tokens(batch_size)

        full_input = torch.cat([prompt_tokens, seq_tokens], dim=1)

        hidden_states = self.forward_backbone(full_input)
        return hidden_states

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
