"""
src/utils.py
Funções utilitárias compartilhadas por todo o pipeline:
- carregamento de configuração YAML
- logging padronizado
- caminhos absolutos baseados na raiz do projeto
"""

import logging
import sys
from pathlib import Path

import yaml

# raiz do projeto = dois níveis acima deste arquivo (src/ -> raiz)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_logger(name: str) -> logging.Logger:
    """Cria um logger padronizado para qualquer módulo do projeto."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carrega o arquivo de configuração YAML do pipeline."""
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(relative_path: str) -> Path:
    """Resolve um caminho relativo à raiz do projeto."""
    return PROJECT_ROOT / relative_path
