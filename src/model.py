import torch
from torch import nn

from src.layers import RMSNorm, SwiGLU, Linear, Embedding
from src.attention import Gqa


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(
            d_model,
            norm_eps=norm_eps,
            device=device,
            dtype=dtype,
        )

        self.attn = Gqa(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            context_length=context_length,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )

        self.ffn_norm = RMSNorm(
            d_model,
            norm_eps=norm_eps,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Pre-norm attention + residual
        x = x + self.attn(
            self.attn_norm(x),
            token_positions,
        )

        # Pre-norm FFN + residual
        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        # Token embedding
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                context_length=context_length,
                rope_theta=rope_theta,
                norm_eps=norm_eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Final RMSNorm
        self.final_norm = RMSNorm(
            d_model=d_model,
            norm_eps=norm_eps,
            device=device,
            dtype=dtype,
        )

        # Untied output projection
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError("sequence length exceeds context length")

        if token_positions is None:
            token_positions = torch.arange(
                sequence_length,
                device=token_ids.device,
                dtype=torch.long,
            )

        # [B, N] -> [B, N, d_model]
        x = self.token_embedding(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.final_norm(x)

        # [B, N, d_model] -> [B, N, vocab_size]
        logits = self.lm_head(x)

        return logits