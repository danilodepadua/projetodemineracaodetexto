from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPOSITORY_ROOT / ".env", override=False)


def _path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value)

    if not path.is_absolute():
        path = REPOSITORY_ROOT / path

    return path


def _value_from_env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    """Repository paths and filenames loaded from .env."""

    model_config_path: Path
    data_dir: Path
    train_labels_filename: str
    valid_labels_filename: str
    train_tfidf_filename: str
    valid_tfidf_filename: str
    models_dir: Path
    evaluation_dir: Path
    model_filename_template: str
    metrics_filename_template: str
    tuning_filename: str

    @property
    def train_labels_path(self) -> Path:
        return self.data_dir / self.train_labels_filename

    @property
    def valid_labels_path(self) -> Path:
        return self.data_dir / self.valid_labels_filename

    @property
    def train_tfidf_path(self) -> Path:
        return self.data_dir / self.train_tfidf_filename

    @property
    def valid_tfidf_path(self) -> Path:
        return self.data_dir / self.valid_tfidf_filename


SETTINGS = Settings(
    model_config_path=_path_from_env(
        "MODEL_CONFIG_PATH",
        "config/models.yaml",
    ),
    data_dir=_path_from_env("DATA_DIR", "data"),
    train_labels_filename=_value_from_env(
        "TRAIN_LABELS_FILENAME",
        "train_limpo.csv",
    ),
    valid_labels_filename=_value_from_env(
        "VALID_LABELS_FILENAME",
        "valid_limpo.csv",
    ),
    train_tfidf_filename=_value_from_env(
        "TRAIN_TFIDF_FILENAME",
        "X_train_tfidf.npz",
    ),
    valid_tfidf_filename=_value_from_env(
        "VALID_TFIDF_FILENAME",
        "X_valid_tfidf.npz",
    ),
    models_dir=_path_from_env("MODELS_DIR", "artifacts/models"),
    evaluation_dir=_path_from_env(
        "EVALUATION_DIR",
        "artifacts/evaluation",
    ),
    model_filename_template=_value_from_env(
        "MODEL_FILENAME_TEMPLATE",
        "{target}.joblib",
    ),
    metrics_filename_template=_value_from_env(
        "METRICS_FILENAME_TEMPLATE",
        "{model}.json",
    ),
    tuning_filename=_value_from_env(
        "TUNING_FILENAME",
        "tuning.json",
    ),
)
