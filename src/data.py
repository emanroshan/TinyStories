from pathlib import Path
import torch
import numpy as np


def load_token_array(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.stat().st_size % 2 != 0:
        raise ValueError("token file must have an even byte length")

    return np.memmap(
        path,
        mode="r",
        dtype=np.dtype("<u2"),
    )




def get_batch(
    dataset,
    batch_size: int,
    sequence_length: int,
    device,
    generator: torch.Generator,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if len(dataset) < sequence_length + 1:
        raise ValueError("dataset is too short")

    starts = torch.randint(
        0,
        len(dataset) - sequence_length,
        (batch_size,),
        generator=generator,
    )

    windows = np.stack(
        [
            dataset[start:start + sequence_length + 1]
            for start in starts.tolist()
        ]
    )

    batch = torch.from_numpy(windows.astype(np.int64))

    x = batch[:, :-1].to(device=device, dtype=torch.long)
    y = batch[:, 1:].to(device=device, dtype=torch.long)

    return x, y