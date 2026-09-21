import numpy as np
from src.models.config import get_model_config, load_config
from src.models.tune import tune_model


def test_model_config_defines_tunable_models_and_parameters():
    config = load_config()

    assert set(config["models"]) == {"ridge", "linear_svr"}
    ridge = get_model_config(config, "ridge")
    assert ridge["params"]["alpha"] == 1.0
    assert ridge["tuning"]["alpha"] == [0.01, 0.1, 1.0, 10.0, 100.0]


def test_tune_model_uses_configured_parameter_grid():
    config = load_config()
    ridge = get_model_config(config, "ridge")
    targets = ["formal_register"]
    y_train = {"formal_register": np.array([1, 2, 3, 4])}
    y_valid = {"formal_register": np.array([2, 3])}
    X_train = np.arange(8, dtype=float).reshape(4, 2)
    X_valid = np.arange(4, dtype=float).reshape(2, 2)

    results = tune_model(
        "ridge",
        ridge,
        y_train,
        y_valid,
        X_train,
        X_valid,
        targets,
    )

    expected_combinations = 1
    for values in ridge["tuning"].values():
        expected_combinations *= len(values)
    assert len(results["formal_register"]) == expected_combinations
    assert {"alpha", "rmse", "mae"} <= set(results["formal_register"][0])
