import torch


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    if logits.ndim < 2:
        raise ValueError("logits must have at least 2 dimensions")

    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            "targets shape must match logits shape excluding final dimension"
        )

    # Stable log-sum-exp
    max_logits = torch.max(
        logits,
        dim=-1,
        keepdim=True,
    ).values

    shifted_logits = logits - max_logits

    log_sum_exp = (
        torch.log(
            torch.sum(
                torch.exp(shifted_logits),
                dim=-1,
            )
        )
        + max_logits.squeeze(-1)
    )

    # Logit corresponding to correct target class
    target_logits = torch.gather(
        logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_sum_exp - target_logits

    return torch.mean(losses)