from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters fixed for the reproducible BiLSTM baseline."""

    seed: int = 42
    vocab_size: int = 30_000
    max_length: int = 400
    embedding_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    # Deliberately tiny verification run. The full experiment will replace only
    # these three split sizes and training schedule after this path is validated.
    max_epochs: int = 1
    patience: int = 1
    train_size: int = 256
    validation_size: int = 64
    test_size: int = 128

    @property
    def output_paths(self) -> dict[str, Path]:
        return {
            "processed": ROOT / "data" / "processed",
            "splits": ROOT / "data" / "split_indices",
            "vocabulary": ROOT / "data" / "vocabulary",
            "models": ROOT / "models",
            "results": ROOT / "results",
        }


def experiment_config(full: bool = False) -> ExperimentConfig:
    """Return the smoke configuration or the report-specified final configuration."""
    if not full:
        return ExperimentConfig()
    return ExperimentConfig(
        train_size=20_000,
        validation_size=5_000,
        test_size=25_000,
        batch_size=128,
        max_epochs=8,
        patience=2,
    )
