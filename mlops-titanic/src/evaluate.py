"""
src/evaluate.py
Responsável por:
1. Avaliar cada modelo treinado com múltiplas métricas
   (accuracy, f1_score, roc_auc, precision, recall)
2. Consolidar os resultados em uma tabela comparativa
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test, y_test) -> dict:
    """Calcula métricas de classificação para um único modelo."""
    y_pred = model.predict(X_test)

    # proba é necessária para ROC-AUC; modelos de classificação binária do
    # sklearn/xgboost expõem predict_proba
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    return metrics


def evaluate_all_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Avalia todos os modelos treinados e retorna um DataFrame comparativo,
    ordenado pela métrica f1_score (apenas para visualização; a escolha
    oficial do campeão é feita em src/model_selection.py).
    """
    results = []
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
        logger.info(f"[{name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "model"))

    df_results = pd.DataFrame(results).set_index("model")
    df_results = df_results.sort_values(by="f1_score", ascending=False)
    return df_results


if __name__ == "__main__":
    from src.train import run_training

    trained_models, X_test, y_test, _, _ = run_training()
    results = evaluate_all_models(trained_models, X_test, y_test)
    print("\n=== COMPARATIVO DE MODELOS ===")
    print(results.round(4))
