import argparse

import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.data import get_batch, load_token_array
from src.loss import cross_entropy
from src.model import TransformerLM
from src.optim import AdamW, get_lr_cosine_schedule, gradient_clipping


def evaluate_validation(
    model,
    val_tokens,
    batch_size,
    sequence_length,
    num_validation_batches,
    device,
    val_generator,
):
    was_training = model.training
    model.eval()
    total_loss = 0.0

    with torch.inference_mode():
        for _ in range(num_validation_batches):
            x, y = get_batch(
                val_tokens,
                batch_size,
                sequence_length,
                device,
                val_generator,
            )
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()

    model.train(was_training)
    return total_loss / num_validation_batches


def train(args):
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if args.num_validation_batches <= 0:
        raise ValueError("num_validation_batches must be positive")

    device = torch.device(args.device)

    train_tokens = load_token_array(args.train_data)
    val_tokens = load_token_array(args.val_data)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_q_heads=args.n_q_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_eps=args.norm_eps,
        device=device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate_max,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.train_seed)

    val_generator = torch.Generator()
    val_generator.manual_seed(args.val_seed)

    next_step = 0

    if args.resume is not None:
        next_step = load_checkpoint(
            args.resume,
            model,
            optimizer,
            train_generator,
            val_generator,
        )

    for step in range(next_step, args.num_steps):
        model.train()

        lr = get_lr_cosine_schedule(
            step,
            args.learning_rate_max,
            args.learning_rate_min,
            args.warmup_steps,
            args.cosine_steps,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        train_loss = 0.0

        for _ in range(args.gradient_accumulation_steps):
            x, y = get_batch(
                train_tokens,
                args.batch_size,
                args.sequence_length,
                device,
                train_generator,
            )

            logits = model(x)
            microbatch_loss = cross_entropy(logits, y)

            (
                microbatch_loss
                / args.gradient_accumulation_steps
            ).backward()

            train_loss += microbatch_loss.detach().item()

        train_loss /= args.gradient_accumulation_steps

        grad_norm = gradient_clipping(
            model.parameters(),
            args.max_grad_norm,
        )

        optimizer.step()

        completed_steps = step + 1
        final_step = completed_steps == args.num_steps

        should_validate = (
            final_step
            or completed_steps % args.eval_interval == 0
        )

        should_log = (
            should_validate
            or completed_steps % args.log_interval == 0
        )

        should_checkpoint = (
            final_step
            or completed_steps % args.checkpoint_interval == 0
        )

        validation_loss = None

        if should_validate:
            validation_loss = evaluate_validation(
                model,
                val_tokens,
                args.batch_size,
                args.sequence_length,
                args.num_validation_batches,
                device,
                val_generator,
            )

        if should_log:
            message = (
                f"step={completed_steps} "
                f"train_loss={train_loss:.6f} "
                f"lr={lr:.8f} "
                f"grad_norm={grad_norm:.6f}"
            )

            if validation_loss is not None:
                message += f" val_loss={validation_loss:.6f}"

            print(message, flush=True)

        if should_checkpoint:
            checkpoint_path = (
                f"{args.checkpoint_path.rsplit('.', 1)[0]}"
                f"_{completed_steps}.pt"
            )

            save_checkpoint(
                model,
                optimizer,
                completed_steps,
                train_generator,
                val_generator,
                checkpoint_path,
            )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--n-q-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--norm-eps", type=float, default=1e-5)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
    )

    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument(
        "--learning-rate-max",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--learning-rate-min",
        type=float,
        default=3e-5,
    )
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--cosine-steps", type=int, default=9999)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num-validation-batches",
        type=int,
        default=100,
    )

    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())