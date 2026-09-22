from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR

from ..data.schema import TARGET_COLUMNS
from ..representations.catalog import REPRESENTATION_NAMES

DEFAULT_CONFIG_PATH = Path("config/models.yaml")

MODEL_FACTORIES = {
    "ridge": Ridge,
    "linear_svr": LinearSVR,
    "random_forest":RandomForestRegressor,
    "decision_tree": DecisionTreeRegressor
}


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the model and tuning configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Model configuration must contain a YAML mapping")

    targets = config.get("targets", TARGET_COLUMNS)
    if targets != TARGET_COLUMNS:
        raise ValueError(
            "Model configuration targets must match the canonical data schema"
        )
    representations = config.get("representations")
    if (
        not isinstance(representations, list)
        or not representations
        or len(representations) != len(set(representations))
        or any(
            not isinstance(name, str) or name not in REPRESENTATION_NAMES
            for name in representations
        )
    ):
        available = ", ".join(REPRESENTATION_NAMES)
        raise ValueError(
            "Model configuration must define unique canonical representations "
            f"from: {available}"
        )

    models = config.get("models")

    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(target, str) and target for target in targets)
    ):
        raise ValueError("Model configuration must define a non-empty targets list")

    if not isinstance(models, dict) or not models:
        raise ValueError("Model configuration must define a non-empty models mapping")

    for model_name, model_config in models.items():
        if model_name not in MODEL_FACTORIES:
            raise ValueError(f"Unsupported model in configuration: {model_name}")

        if not isinstance(model_config, dict):
            raise ValueError(f"Configuration for {model_name} must be a mapping")

        params = model_config.get("params")

        if not isinstance(params, dict) or not params:
            raise ValueError(f"Parameters for {model_name} must be a non-empty mapping")

        for parameter_name, parameter in params.items():
            if isinstance(parameter, dict):
                values = parameter.get("values")

                if "default" not in parameter or "values" not in parameter:
                    raise ValueError(
                        f"Tunable parameter {model_name}.{parameter_name} "
                        "must define default and values"
                    )

                if not isinstance(values, list) or not values:
                    raise ValueError(
                        f"Tuning values for {model_name}.{parameter_name} "
                        "must be a non-empty list"
                    )

                if parameter["default"] not in values:
                    raise ValueError(
                        f"Default value for {model_name}.{parameter_name} "
                        "must be included in values"
                    )

            elif parameter is None:
                raise ValueError(
                    f"Parameter {model_name}.{parameter_name} must define a value"
                )

    return config


def get_model_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Return the configuration for a named model."""
    models = config["models"]

    if model_name not in models:
        available = ", ".join(models)
        raise ValueError(
            f"Unsupported model: {model_name}. Available models: {available}"
        )

    raw_model_config = models[model_name]
    params = {}
    tuning = {}

    for parameter_name, parameter in raw_model_config["params"].items():
        if isinstance(parameter, dict):
            params[parameter_name] = parameter["default"]
            tuning[parameter_name] = parameter["values"]
        else:
            params[parameter_name] = parameter

    return {
        **raw_model_config,
        "params": params,
        "tuning": tuning,
    }


def create_model(model_name: str, params: dict[str, Any]):
    """Create an unfitted model from YAML parameters."""
    model_params = dict(params)

    if model_name == "linear_svr":
        dual = model_params.get("dual", "auto")

        if isinstance(dual, str):
            model_params["dual"] = {
                "true": True,
                "false": False,
                "auto": "auto",
            }.get(dual.lower(), dual)

    try:
        model_factory = MODEL_FACTORIES[model_name]
    except KeyError as error:
        available = ", ".join(MODEL_FACTORIES)
        raise ValueError(
            f"Unsupported model: {model_name}. Available models: {available}"
        ) from error

    return model_factory(**model_params)
