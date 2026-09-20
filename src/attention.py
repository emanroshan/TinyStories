import math
import torch
from torch import nn
from src.layers import Linear
from src.rope import Rope

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax."""
    max_value = torch.max(x, dim=dim, keepdim=True).values
    shifted = x - max_value
    exp_x = torch.exp(shifted)

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention.

    True in mask  = allowed
    False in mask = blocked
    """

    # Q and K must have the same feature dimension d_k
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(
            "query and key feature dimensions must match"
        )

    # K and V must have the same sequence length
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError(
            "key and value sequence lengths must match"
        )

    # Mask must be boolean
    if mask is not None and mask.dtype != torch.bool:
        raise TypeError(
            "mask must be boolean"
        )

    d_k = queries.shape[-1]

    # QK^T
    scores = torch.matmul(
        queries,
        keys.transpose(-2, -1),
    )

    # Scale
    scores = scores / math.sqrt(d_k)

    # True = allowed, False = blocked
    if mask is not None:
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    # Softmax over keys
    attention_weights = softmax(
        scores,
        dim=-1,
    )

    # Weighted sum of values
    output = torch.matmul(
        attention_weights,
        values,
    )

    return output


class Gqa(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        context_length: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if d_model % n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")

        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.group_size = n_q_heads // n_kv_heads

        self.q_proj = Linear(
            d_model,
            n_q_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )

        self.k_proj = Linear(
            d_model,
            n_kv_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )

        self.v_proj = Linear(
            d_model,
            n_kv_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )

        self.o_proj = Linear(
            n_q_heads * self.head_dim,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.rope = Rope(
            head_dim=self.head_dim,
            rope_theta=rope_theta,
            context_length=context_length,
            device=device,
        )

    # IMPORTANT:
    # forward is indented INSIDE Gqa
    # at the SAME indentation level as __init__
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        B, N, _ = x.shape

        if token_positions is None:
            token_positions = torch.arange(
                N,
                device=x.device,
                dtype=torch.long,
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Q: [B, N, hq, dh] -> [B, hq, N, dh]
        q = q.reshape(
            B,
            N,
            self.n_q_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        # K: [B, N, hkv, dh] -> [B, hkv, N, dh]
        k = k.reshape(
            B,
            N,
            self.n_kv_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        # V: [B, N, hkv, dh] -> [B, hkv, N, dh]
        v = v.reshape(
            B,
            N,
            self.n_kv_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        # RoPE only Q and K
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # Group query heads:
        # [B, hq, N, dh]
        # ->
        # [B, hkv, group_size, N, dh]
        q = q.reshape(
            B,
            self.n_kv_heads,
            self.group_size,
            N,
            self.head_dim,
        )

        # Add group dimension to K/V.
        # Broadcasting instead of repeating.
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)

        # causal mask
        causal_mask = torch.tril(
            torch.ones(
                N,
                N,
                dtype=torch.bool,
                device=x.device,
            )
        )

        attended = scaled_dot_product_attention(
            q,
            k,
            v,
            causal_mask,
        )

        # [B,hkv,g,N,dh]
        # -> [B,hq,N,dh]
        attended = attended.reshape(
            B,
            self.n_q_heads,
            N,
            self.head_dim,
        )

        # -> [B,N,hq,dh]
        attended = attended.permute(0, 2, 1, 3)

        # -> [B,N,d_model]
        attended = attended.reshape(
            B,
            N,
            self.d_model,
        )

        return self.o_proj(attended)