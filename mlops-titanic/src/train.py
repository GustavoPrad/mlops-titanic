"""
src/train.py
Responsável por:
1. Split treino/teste
2. Pré-processamento (encoding + scaling) via Pipeline do sklearn
3. Treinamento de 3 modelos: Logistic Regression, Random Forest, XGBoost
4. Retorno dos modelos treinados + dados de teste para avaliação
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_processing import run_data_pipeline
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def build_preprocessor(config: dict) -> ColumnTransformer:
    """
    Cria o pré-processador de features:
    - Numéricas: padronização (StandardScaler)
    - Categóricas: one-hot encoding
    """
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def split_data(df: pd.DataFrame, config: dict):
    """Separa features/target e faz o split treino/teste."""
    target_col = config["data"]["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    logger.info(f"Split realizado: treino={X_train.shape[0]}, teste={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def get_model_definitions(config: dict) -> dict:
    """
    Define os 3 modelos de classificação a partir do config.yaml.
    Cada modelo é embrulhado em um Pipeline com o pré-processador,
    garantindo que o mesmo fluxo de transformação seja usado em
    treino e em inferência (evita data leakage e inconsistência).
    """
    models_cfg = config["models"]
    preprocessor = build_preprocessor(config)
    models = {}

    if models_cfg["logistic_regression"]["enabled"]:
        params = models_cfg["logistic_regression"]["params"]
        models["logistic_regression"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(**params)),
            ]
        )

    if models_cfg["random_forest"]["enabled"]:
        params = models_cfg["random_forest"]["params"]
        models["random_forest"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(**params)),
            ]
        )

    if models_cfg["xgboost"]["enabled"]:
        params = models_cfg["xgboost"]["params"]
        models["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", XGBClassifier(**params)),
            ]
        )

    logger.info(f"Modelos habilitados para treino: {list(models.keys())}")
    return models


def train_all_models(X_train, y_train, config: dict) -> dict:
    """Treina todos os modelos habilitados e retorna o dicionário {nome: modelo_treinado}."""
    models = get_model_definitions(config)
    trained_models = {}

    for name, pipeline in models.items():
        logger.info(f"Treinando modelo: {name}")
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        logger.info(f"Modelo {name} treinado com sucesso")

    return trained_models


def run_training(config: dict = None):
    """Orquestra: carrega dados processados -> split -> treina os 3 modelos."""
    if config is None:
        config = load_config()

    df = run_data_pipeline(config)
    X_train, X_test, y_train, y_test = split_data(df, config)
    trained_models = train_all_models(X_train, y_train, config)

    return trained_models, X_test, y_test, X_train, y_train


if __name__ == "__main__":
    run_training()
