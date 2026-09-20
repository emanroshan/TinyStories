import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self._validate_hyperparameters(lr, betas, eps, weight_decay)

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

        for group in self.param_groups:
            self._validate_hyperparameters(
                group["lr"],
                group["betas"],
                group["eps"],
                group["weight_decay"],
            )

    @staticmethod
    def _validate_hyperparameters(lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError("learning rate must be non-negative")

        if eps < 0:
            raise ValueError("epsilon must be non-negative")

        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        beta1, beta2 = betas

        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must satisfy 0 <= beta1 < 1")

        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must satisfy 0 <= beta2 < 1")

    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                self._validate_hyperparameters(
                    lr,
                    (beta1, beta2),
                    eps,
                    weight_decay,
                )

                for p in group["params"]:
                    grad = p.grad

                    if grad is None:
                        continue

                    if grad.is_sparse:
                        raise RuntimeError("AdamW does not support sparse gradients")

                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    state["step"] += 1
                    t = state["step"]

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad,
                        grad,
                        value=1 - beta2,
                    )

                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t

                    exp_avg_hat = exp_avg / bias_correction1
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2

                    p.mul_(1 - lr * weight_decay)

                    p.addcdiv_(
                        exp_avg_hat,
                        exp_avg_sq_hat.sqrt().add(eps),
                        value=-lr,
                    )

        return loss




def get_lr_cosine_schedule(
    step: int,
    learning_rate_max: float,
    learning_rate_min: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError("step must be an integer")

    if step < 0:
        raise ValueError("step must be non-negative")

    if learning_rate_min < 0 or learning_rate_min > learning_rate_max:
        raise ValueError("learning rates must satisfy 0 <= min <= max")

    if warmup_steps < 0 or warmup_steps >= cosine_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < cosine_steps")

    if step < warmup_steps:
        return (step / warmup_steps) * learning_rate_max

    if step <= cosine_steps:
        progress = (step - warmup_steps) / (cosine_steps - warmup_steps)

        return learning_rate_min + 0.5 * (
            1 + math.cos(math.pi * progress)
        ) * (learning_rate_max - learning_rate_min)

    return learning_rate_min

def gradient_clipping(parameters, max_l2_norm: float) -> float:
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return 0.0

    total_squared = 0.0

    for grad in grads:
        total_squared += float(torch.sum(grad.detach() ** 2))

    total_norm = total_squared ** 0.5

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        with torch.no_grad():
            for grad in grads:
                grad.mul_(scale)

    return float(total_norm)