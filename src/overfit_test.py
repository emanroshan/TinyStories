import torch

from src.data import load_token_array, get_batch
from src.model import TransformerLM
from src.loss import cross_entropy
from src.optim import AdamW, gradient_clipping


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

train_tokens = load_token_array(
    "data/tinystories/data/train.bin"
)

generator = torch.Generator()
generator.manual_seed(0)

x, y = get_batch(
    train_tokens,
    batch_size=4,
    sequence_length=64,
    device=device,
    generator=generator,
)

model = TransformerLM(
    vocab_size=8192,
    context_length=64,
    d_model=128,
    num_layers=2,
    n_q_heads=4,
    n_kv_heads=2,
    d_ff=352,
    rope_theta=10000.0,
    norm_eps=1e-5,
    device=device,
)

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)

model.train()

for step in range(101):
    optimizer.zero_grad()

    logits = model(x)
    loss = cross_entropy(logits, y)

    loss.backward()

    gradient_clipping(
        model.parameters(),
        1.0,
    )

    optimizer.step()

    if step % 10 == 0:
        print(
            f"step={step} "
            f"loss={loss.item():.6f}"
        )