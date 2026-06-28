"""
src/pipeline.py
ORQUESTRADOR CENTRAL DO PIPELINE DE MLOPS.

Executa o fluxo completo:
1. Ingestão + limpeza de dados
2. Treinamento de 3 modelos
3. Avaliação com múltiplas métricas
4. Seleção automática do melhor modelo (com desempate)
5. Decisão de promoção (substitui o campeão atual? ciclo de retraining)
6. Persistência do modelo vencedor + registro de versão

Uso:
    python -m src.pipeline                  # treino + avaliação completos
    python -m src.pipeline --force-promote  # ignora comparação com campeão atual
"""

import argparse

from src.data_processing import run_data_pipeline
from src.evaluate import evaluate_all_models
from src.model_selection import is_new_model_better, select_best_model
from src.persistence import load_champion_metadata, save_champion
from src.train import run_training
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def run_pipeline(config: dict = None, force_promote: bool = False) -> dict:
    """
    Executa o pipeline de MLOps de ponta a ponta e retorna um resumo
    da execução (modelo escolhido, métricas, se foi promovido, etc).
    """
    if config is None:
        config = load_config()

    logger.info("========== INÍCIO DO PIPELINE DE MLOPS ==========")

    # 1. Dados
    logger.info("[1/5] Ingestão e processamento de dados")
    run_data_pipeline(config)

    # 2. Treino dos 3 modelos
    logger.info("[2/5] Treinamento dos modelos")
    trained_models, X_test, y_test, X_train, y_train = run_training(config)

    # 3. Avaliação
    logger.info("[3/5] Avaliação dos modelos")
    results_df = evaluate_all_models(trained_models, X_test, y_test)
    print("\n=== COMPARATIVO DE MODELOS ===")
    print(results_df.round(4))

    # 4. Seleção automática do melhor modelo
    logger.info("[4/5] Seleção automática do melhor modelo")
    best_model_name, best_metrics = select_best_model(results_df, config)
    best_model = trained_models[best_model_name]

    # 5. Decide promoção (ciclo de retraining automatizado)
    logger.info("[5/5] Avaliação de promoção e persistência")
    current_champion_metadata = load_champion_metadata(config)
    current_champion_metrics = (
        current_champion_metadata["metrics"] if current_champion_metadata else None
    )

    should_promote = force_promote or is_new_model_better(
        best_metrics, current_champion_metrics, config
    )

    metadata = save_champion(
        model=best_model,
        model_name=best_model_name,
        metrics=best_metrics,
        config=config,
        promoted=should_promote,
    )

    logger.info("========== FIM DO PIPELINE DE MLOPS ==========")

    summary = {
        "results_table": results_df,
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "promoted": should_promote,
        "metadata": metadata,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Pipeline de MLOps - Titanic")
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Promove o melhor modelo do batch atual independentemente do campeão existente",
    )
    args = parser.parse_args()

    summary = run_pipeline(force_promote=args.force_promote)

    print("\n=== RESUMO DA EXECUÇÃO ===")
    print(f"Melhor modelo do batch: {summary['best_model_name']}")
    print(f"Promovido a campeão:    {summary['promoted']}")
    print(f"Versão registrada:      {summary['metadata']['version']}")


if __name__ == "__main__":
    main()
