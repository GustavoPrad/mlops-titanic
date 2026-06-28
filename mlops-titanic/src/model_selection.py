"""
src/model_selection.py
Responsável por:
1. Comparar os modelos treinados com base em métricas
2. Escolher automaticamente o melhor modelo (métrica principal)
3. Usar métrica de desempate em caso de empate
4. Decidir se o novo modelo deve substituir o campeão atual
   (lógica usada no ciclo automatizado de retraining)
"""

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def select_best_model(results_df: pd.DataFrame, config: dict) -> tuple[str, pd.Series]:
    """
    Seleciona o melhor modelo a partir da tabela de métricas.

    Critério:
    1. Ordena pela métrica principal (ex: f1_score) de forma decrescente.
    2. Em caso de empate exato na métrica principal, usa a métrica de
       desempate (ex: roc_auc) também de forma decrescente.

    Retorna: (nome_do_modelo, série_com_as_métricas)
    """
    primary = config["evaluation"]["primary_metric"]
    tiebreaker = config["evaluation"]["tiebreaker_metric"]

    sorted_df = results_df.sort_values(
        by=[primary, tiebreaker], ascending=[False, False]
    )

    best_model_name = sorted_df.index[0]
    best_metrics = sorted_df.iloc[0]

    # checagem explícita de empate (mais de um modelo com o mesmo valor
    # na métrica principal, arredondado para evitar ruído de float)
    top_primary_value = round(sorted_df.iloc[0][primary], 6)
    tied_models = sorted_df[round(sorted_df[primary], 6) == top_primary_value]

    if len(tied_models) > 1:
        logger.info(
            f"Empate detectado na métrica '{primary}' entre: "
            f"{list(tied_models.index)}. Desempate por '{tiebreaker}'."
        )

    logger.info(
        f"Modelo selecionado: {best_model_name} "
        f"({primary}={best_metrics[primary]:.4f}, {tiebreaker}={best_metrics[tiebreaker]:.4f})"
    )

    return best_model_name, best_metrics


def is_new_model_better(
    new_metrics: pd.Series, current_champion_metrics: dict | None, config: dict
) -> bool:
    """
    Decide se um modelo recém-treinado deve substituir o campeão atual.
    Usado no ciclo automatizado (retraining contínuo).

    Se não houver campeão atual, o novo modelo vence por padrão.
    Caso contrário, compara a métrica principal com uma margem mínima
    de melhoria configurável (evita troca por ruído estatístico).
    """
    primary = config["evaluation"]["primary_metric"]
    threshold = config["evaluation"].get("improvement_threshold", 0.0)

    if current_champion_metrics is None:
        logger.info("Nenhum campeão anterior encontrado. Novo modelo será promovido.")
        return True

    new_score = new_metrics[primary]
    current_score = current_champion_metrics[primary]

    improved = new_score > (current_score + threshold)

    if improved:
        logger.info(
            f"Novo modelo melhora '{primary}': {current_score:.4f} -> {new_score:.4f}. "
            f"Promovendo novo campeão."
        )
    else:
        logger.info(
            f"Novo modelo NÃO supera o campeão atual em '{primary}': "
            f"{new_score:.4f} <= {current_score:.4f} (+ threshold {threshold}). "
            f"Mantendo campeão atual."
        )

    return improved
