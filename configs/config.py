from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 13
    batch_size: int = 32
    buffer_size: int = 1000
    lr: float = 1e-3
    epochs: int = 10
    num_classes: int = 10