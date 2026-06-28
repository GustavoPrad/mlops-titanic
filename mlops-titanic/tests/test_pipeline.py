"""
tests/test_pipeline.py
Testes automatizados do pipeline de MLOps.
Cobrem: processamento de dados, treino, avaliação, seleção e persistência.

Rodar com: pytest tests/ -v
"""

import shutil

import pandas as pd
import pytest

from src.data_processing import clean_and_engineer, run_data_pipeline
from src.evaluate import evaluate_all_models, evaluate_model
from src.model_selection import is_new_model_better, select_best_model
from src.train import run_training
from src.utils import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def trained_pipeline_outputs(config):
    """Roda o treino uma única vez e reaproveita entre os testes (mais rápido)."""
    trained_models, X_test, y_test, X_train, y_train = run_training(config)
    return trained_models, X_test, y_test


def test_config_loads_successfully(config):
    assert "data" in config
    assert "models" in config
    assert "evaluation" in config


def test_data_pipeline_runs(config):
    df = run_data_pipeline(config)
    assert df.shape[0] > 0
    assert config["data"]["target_column"] in df.columns
    # não deve haver valores faltantes após a limpeza
    assert df["Age"].isna().sum() == 0
    assert df["Fare"].isna().sum() == 0


def test_family_size_feature_created(config):
    df_raw = pd.read_csv(config["data"]["raw_path"].replace("data/raw/titanic.csv", "data/raw/titanic.csv"))
    df_clean = clean_and_engineer(df_raw, config)
    assert "FamilySize" in df_clean.columns
    assert (df_clean["FamilySize"] == df_clean["SibSp"] + df_clean["Parch"] + 1).all()


def test_three_models_are_trained(trained_pipeline_outputs):
    trained_models, _, _ = trained_pipeline_outputs
    assert len(trained_models) == 3
    assert "logistic_regression" in trained_models
    assert "random_forest" in trained_models
    assert "xgboost" in trained_models


def test_models_produce_valid_predictions(trained_pipeline_outputs):
    trained_models, X_test, y_test = trained_pipeline_outputs
    for name, model in trained_models.items():
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1}), f"Modelo {name} retornou classes inválidas"


def test_evaluate_model_returns_expected_metrics(trained_pipeline_outputs):
    trained_models, X_test, y_test = trained_pipeline_outputs
    model = trained_models["random_forest"]
    metrics = evaluate_model(model, X_test, y_test)

    expected_keys = {"accuracy", "f1_score", "roc_auc", "precision", "recall"}
    assert expected_keys.issubset(metrics.keys())
    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"Métrica {key} fora do intervalo [0,1]: {value}"


def test_best_model_selection(trained_pipeline_outputs, config):
    trained_models, X_test, y_test = trained_pipeline_outputs
    results_df = evaluate_all_models(trained_models, X_test, y_test)
    best_name, best_metrics = select_best_model(results_df, config)

    primary_metric = config["evaluation"]["primary_metric"]
    # o modelo escolhido deve realmente ter a maior métrica principal
    assert best_metrics[primary_metric] == results_df[primary_metric].max()
    assert best_name in trained_models


def test_promotion_logic_no_previous_champion(config):
    fake_metrics = pd.Series({"f1_score": 0.75, "roc_auc": 0.80})
    result = is_new_model_better(fake_metrics, None, config)
    assert result == True


def test_promotion_logic_improvement(config):
    fake_new = pd.Series({"f1_score": 0.80, "roc_auc": 0.85})
    fake_current = {"f1_score": 0.75, "roc_auc": 0.80}
    assert is_new_model_better(fake_new, fake_current, config) == True


def test_promotion_logic_no_improvement(config):
    fake_new = pd.Series({"f1_score": 0.70, "roc_auc": 0.80})
    fake_current = {"f1_score": 0.75, "roc_auc": 0.80}
    assert is_new_model_better(fake_new, fake_current, config) == False
