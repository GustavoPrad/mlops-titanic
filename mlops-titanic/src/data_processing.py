"""
src/data_processing.py
Responsável por:
1. Importar a base de dados (CSV)
2. Limpar dados (valores faltantes, colunas inúteis)
3. Engenharia de features simples
4. Persistir versão processada em data/processed/
"""

import pandas as pd

from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def load_raw_data(config: dict) -> pd.DataFrame:
    """Importa o CSV bruto do Titanic."""
    raw_path = resolve_path(config["data"]["raw_path"])
    logger.info(f"Carregando dados brutos de {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def clean_and_engineer(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Limpeza básica + feature engineering:
    - Idade: imputação pela mediana (por classe/sexo seria mais refinado,
      mas mediana global mantém o pipeline simples e reprodutível)
    - Embarked: imputação pela moda
    - Fare: imputação pela mediana
    - FamilySize = SibSp + Parch + 1 (feature derivada)
    - Remoção de colunas de alta cardinalidade / sem valor preditivo direto
    """
    df = df.copy()

    # Feature derivada: tamanho da família a bordo
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Imputação de valores faltantes
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if df["Embarked"].isna().any():
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Remove colunas que não entram no modelo (definidas no config)
    drop_cols = [c for c in config["data"]["drop_columns"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    logger.info(f"Dados limpos e enriquecidos. Shape final: {df.shape}")
    return df


def save_processed_data(df: pd.DataFrame, config: dict) -> None:
    """Salva a versão processada em disco (rastreabilidade do pipeline)."""
    processed_path = resolve_path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Dados processados salvos em {processed_path}")


def run_data_pipeline(config: dict = None) -> pd.DataFrame:
    """Orquestra a etapa completa de ingestão + limpeza + persistência."""
    if config is None:
        config = load_config()
    df_raw = load_raw_data(config)
    df_clean = clean_and_engineer(df_raw, config)
    save_processed_data(df_clean, config)
    return df_clean


if __name__ == "__main__":
    run_data_pipeline()
