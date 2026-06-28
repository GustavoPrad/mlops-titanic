"""
src/auto_retrain.py
CICLO AUTOMATIZADO DE MLOPS.

Simula o fluxo de produção descrito no requisito 7:
"Quando novos dados estiverem disponíveis, o pipeline roda automaticamente,
o modelo é retreinado, modelos são reavaliados, e se houver melhoria,
o modelo novo substitui o antigo."

Duas formas de uso:

1) MODO WATCH (contínuo, recomendado para demonstração):
   Observa a pasta data/incoming/ usando watchdog. Sempre que um novo
   arquivo .csv for adicionado (simulando novos dados de produção),
   ele é concatenado à base de dados raw e o pipeline completo é
   automaticamente re-executado.

       python -m src.auto_retrain --watch

2) MODO ONE-SHOT (para uso em cron job / CI / agendador externo):
   Processa qualquer arquivo pendente em data/incoming/ uma única vez
   e finaliza. Ideal para ser chamado periodicamente por um agendador
   (cron, Airflow, GitHub Actions, etc).

       python -m src.auto_retrain --once
"""

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.pipeline import run_pipeline
from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)

INCOMING_DIR = resolve_path("data/incoming")
PROCESSED_INCOMING_DIR = resolve_path("data/incoming/_processed")


def _merge_new_data_into_raw(new_csv_path: Path, config: dict) -> None:
    """
    Concatena o novo lote de dados (simulando dados de produção)
    à base raw existente, mantendo o histórico completo.
    """
    raw_path = resolve_path(config["data"]["raw_path"])

    df_existing = pd.read_csv(raw_path)
    df_new = pd.read_csv(new_csv_path)

    # garante que apenas colunas conhecidas sejam incorporadas
    common_cols = [c for c in df_existing.columns if c in df_new.columns]
    df_new = df_new[common_cols]

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(raw_path, index=False)

    logger.info(
        f"Novos dados incorporados: +{len(df_new)} linhas "
        f"(total agora: {len(df_combined)} linhas)"
    )


def process_new_file(csv_path: Path):
    """
    Processa um novo arquivo de dados:
    1. Mescla com a base raw
    2. Roda o pipeline completo (treino + avaliação + seleção + persistência)
    3. Move o arquivo processado para evitar reprocessamento
    """
    logger.info(f"Novo arquivo de dados detectado: {csv_path.name}")
    config = load_config()

    try:
        _merge_new_data_into_raw(csv_path, config)
    except Exception as e:
        logger.error(f"Falha ao mesclar novos dados: {e}")
        return

    logger.info("Disparando pipeline automático de retraining...")
    summary = run_pipeline(config)

    PROCESSED_INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(csv_path), str(PROCESSED_INCOMING_DIR / csv_path.name))

    status = "PROMOVIDO a novo campeão" if summary["promoted"] else "mantido (sem melhoria)"
    logger.info(
        f"Ciclo de retraining concluído. Modelo {summary['best_model_name']} {status}. "
        f"Versão registrada: {summary['metadata']['version']}"
    )


def run_once():
    """Processa todos os arquivos CSV pendentes em data/incoming/ uma única vez."""
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    pending_files = sorted(INCOMING_DIR.glob("*.csv"))

    if not pending_files:
        logger.info("Nenhum arquivo novo encontrado em data/incoming/.")
        return

    for csv_path in pending_files:
        process_new_file(csv_path)


class _NewDataHandler(FileSystemEventHandler):
    """Handler do watchdog: reage à criação de novos arquivos .csv."""

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".csv":
            return
        # pequena espera para garantir que o arquivo terminou de ser escrito
        time.sleep(1)
        process_new_file(path)


def run_watch():
    """Inicia o monitoramento contínuo da pasta data/incoming/."""
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Monitorando {INCOMING_DIR} por novos arquivos de dados (CTRL+C para parar)...")

    event_handler = _NewDataHandler()
    observer = Observer()
    observer.schedule(event_handler, str(INCOMING_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Monitoramento interrompido pelo usuário.")
    observer.join()


def main():
    parser = argparse.ArgumentParser(description="Ciclo automatizado de retraining - MLOps Titanic")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--watch", action="store_true", help="Monitora data/incoming/ continuamente")
    group.add_argument("--once", action="store_true", help="Processa arquivos pendentes uma única vez")
    args = parser.parse_args()

    if args.watch:
        run_watch()
    elif args.once:
        run_once()


if __name__ == "__main__":
    main()
