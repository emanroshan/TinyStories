import torch
from torch import nn


class Rope(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        context_length: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.context_length = context_length

        pair_indices = torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
            device=device,
        )

        inv_freq = rope_theta ** (-pair_indices / head_dim)

        positions = torch.arange(
            context_length,
            dtype=torch.float32,
            device=device,
        )

        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer(
            "cos_cache",
            torch.cos(angles),
            persistent=False,
        )

        self.register_buffer(
            "sin_cache",
            torch.sin(angles),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        if x.shape[-1] != self.head_dim:
            raise ValueError("Last dimension must equal head_dim")

        # Positions must contain integer indices.
        if token_positions.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("token_positions must have an integer dtype")

        if token_positions.numel() == 0:
            return x

        if token_positions.min().item() < 0:
            raise ValueError("token positions must be non-negative")

        if token_positions.max().item() >= self.context_length:
            raise ValueError("token position is outside the context length")

        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # Broadcast position dimensions across leading/head dimensions.
        # Broadcast position dimensions across head dimensions.
        if token_positions.ndim == 1:
    # [N, d/2] -> [1, ..., 1, N, d/2]
            while cos.ndim < x.ndim:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
        else:
    # [B, N, d/2] -> [B, 1, ..., 1, N, d/2]
            while cos.ndim < x.ndim:
                cos = cos.unsqueeze(-3)
                sin = sin.unsqueeze(-3)

        cos = cos.to(
            device=x.device,
            dtype=x.dtype,
        )

        sin = sin.to(
            device=x.device,
            dtype=x.dtype,
        )

        # Adjacent pairs:
        #
        # x0,x1
        # x2,x3
        # x4,x5
        # ...
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # [a,b] ->
        # [a cosθ - b sinθ,
        #  a sinθ + b cosθ]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        output = torch.empty_like(x)

        output[..., 0::2] = rotated_even
        output[..., 1::2] = rotated_odd

        return output