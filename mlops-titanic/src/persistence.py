"""
src/persistence.py
Responsável por:
1. Salvar o modelo campeão em disco (.pkl)
2. Salvar metadados do campeão (métricas, timestamp, versão)
3. Manter um "model registry" (histórico de versões) em JSON,
   simulando um registro de modelos (Model Registry) de MLOps
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.utils import get_logger, resolve_path

logger = get_logger(__name__)


def _load_registry(registry_path) -> list:
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Arquivo de registro corrompido em {registry_path}. "
                f"Iniciando um novo histórico."
            )
            return []
    return []


def _next_version(registry: list) -> int:
    if not registry:
        return 1
    return max(entry["version"] for entry in registry) + 1


def load_champion_metadata(config: dict) -> dict | None:
    """Carrega os metadados do campeão atual, se existir."""
    metadata_path = resolve_path(config["persistence"]["champion_metadata_path"])
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_champion(
    model,
    model_name: str,
    metrics: pd.Series,
    config: dict,
    promoted: bool = True,
) -> dict:
    """
    Salva o modelo campeão (.pkl) + metadados (.json) e registra a
    nova versão no histórico (model_registry.json).

    Retorna o dicionário de metadados salvo.
    """
    models_dir = resolve_path(config["persistence"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    champion_path = resolve_path(config["persistence"]["champion_model_path"])
    metadata_path = resolve_path(config["persistence"]["champion_metadata_path"])
    registry_path = resolve_path(config["persistence"]["registry_path"])

    registry = _load_registry(registry_path)
    version = _next_version(registry)

    metadata = {
        "version": version,
        "model_name": model_name,
        "metrics": {k: float(v) for k, v in metrics.items() if k != "model"},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "promoted": bool(promoted),
    }

    if promoted:
        # salva o artefato do modelo (reutilizável para inferência)
        joblib.dump(model, champion_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Modelo campeão salvo em {champion_path} (versão {version})")
    else:
        logger.info(
            f"Modelo {model_name} (versão candidata {version}) NÃO promovido. "
            f"Campeão atual mantido."
        )

    # versão sempre entra no histórico, promovida ou não, para
    # rastreabilidade completa do ciclo de retraining.
    # Escrita atômica (tmp + replace) evita corromper o arquivo
    # caso o processo seja interrompido no meio da escrita.
    registry.append(metadata)
    tmp_path = registry_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    tmp_path.replace(registry_path)

    return metadata


def load_champion_model(config: dict):
    """Carrega o modelo campeão salvo em disco para inferência."""
    champion_path = resolve_path(config["persistence"]["champion_model_path"])
    if not champion_path.exists():
        raise FileNotFoundError(
            f"Nenhum modelo campeão encontrado em {champion_path}. "
            f"Execute o pipeline de treino primeiro (src/pipeline.py)."
        )
    return joblib.load(champion_path)
