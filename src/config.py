from dataclasses import dataclass

@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    num_classes: int = 10