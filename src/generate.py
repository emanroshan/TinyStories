import torch


def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be one-dimensional")

    if prompt_ids.dtype != torch.long:
        raise TypeError("prompt_ids must have dtype torch.long")

    if prompt_ids.numel() == 0:
        raise ValueError("prompt_ids must be non-empty")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    if context_length <= 0:
        raise ValueError("context_length must be positive")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must satisfy 0 < top_p <= 1")

    if not hasattr(model, "context_length"):
        raise ValueError("model must expose context_length")

    if context_length != model.context_length:
        raise ValueError("context_length must equal model.context_length")

    if max_new_tokens == 0:
        return prompt_ids

    was_training = model.training
    model.eval()

    output = prompt_ids.clone()

    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = output[-context_length:]
                logits = model(context.unsqueeze(0))[0, -1]
                logits = logits / temperature

                max_logit = torch.max(logits)
                exp_logits = torch.exp(logits - max_logit)
                probabilities = exp_logits / exp_logits.sum()

                sorted_probs, sorted_indices = torch.sort(
                    probabilities,
                    descending=True,
                )

                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                reached = torch.nonzero(
                    cumulative_probs >= top_p,
                    as_tuple=False,
                )

                if reached.numel() == 0:
                    cutoff = sorted_probs.numel()
                else:
                    cutoff = int(reached[0].item()) + 1

                filtered_probs = sorted_probs[:cutoff]
                filtered_indices = sorted_indices[:cutoff]

                filtered_probs = (
                    filtered_probs / filtered_probs.sum()
                )

                sampled_rank = torch.multinomial(
                    filtered_probs,
                    1,
                    generator=generator,
                )

                next_token = filtered_indices[sampled_rank]

                output = torch.cat(
                    (output, next_token),
                    dim=0,
                )

                if (
                    eot_token_id is not None
                    and next_token.item() == eot_token_id
                ):
                    break
    finally:
        model.train(was_training)

    return output