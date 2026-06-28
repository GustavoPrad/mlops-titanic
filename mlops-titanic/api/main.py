"""
api/main.py
Serviço de inferência (API) do projeto de MLOps.

Endpoints:
- GET  /health           -> healthcheck simples
- GET  /model/info       -> metadados do modelo campeão em produção
- POST /predict          -> recebe dados de um passageiro e retorna a previsão
- POST /model/reload     -> recarrega o modelo do disco (útil após retraining)

O modelo carregado é sempre o último "campeão" salvo por src/pipeline.py.

Para rodar localmente:
    uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthCheck, ModelInfo, PassengerInput, PredictionOutput
from src.persistence import load_champion_metadata, load_champion_model
from src.utils import get_logger, load_config

logger = get_logger(__name__)

# Estado em memória do serviço: modelo carregado + config + metadados
_state = {"model": None, "config": None, "metadata": None}


def _load_model_into_state():
    config = load_config()
    try:
        model = load_champion_model(config)
        metadata = load_champion_metadata(config)
        _state["model"] = model
        _state["config"] = config
        _state["metadata"] = metadata
        logger.info(
            f"Modelo campeão carregado: {metadata['model_name']} "
            f"(versão {metadata['version']})"
        )
    except FileNotFoundError as e:
        logger.warning(str(e))
        _state["model"] = None
        _state["config"] = config
        _state["metadata"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo campeão ao iniciar a API (startup) e libera ao encerrar."""
    _load_model_into_state()
    yield
    _state["model"] = None


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API de inferência do pipeline de MLOps - Titanic Survival Classifier",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS liberado para permitir consumo por front-ends locais (Streamlit, etc)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_feature_row(passenger: PassengerInput) -> pd.DataFrame:
    """
    Converte o input da API no mesmo formato de colunas usado no
    treinamento (incluindo a feature derivada FamilySize).
    """
    family_size = passenger.SibSp + passenger.Parch + 1
    row = {
        "Pclass": passenger.Pclass,
        "Sex": passenger.Sex,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Embarked": passenger.Embarked,
        "FamilySize": family_size,
    }
    return pd.DataFrame([row])


@app.get("/health", response_model=HealthCheck)
def health_check():
    """Verifica se a API está no ar e se há um modelo carregado."""
    model_loaded = _state["model"] is not None
    version = _state["metadata"]["version"] if model_loaded else None
    return HealthCheck(status="ok", model_loaded=model_loaded, model_version=version)


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    """Retorna os metadados do modelo campeão atualmente carregado."""
    if _state["metadata"] is None:
        raise HTTPException(status_code=404, detail="Nenhum modelo campeão carregado.")
    return ModelInfo(**_state["metadata"])


@app.post("/model/reload", response_model=ModelInfo)
def reload_model():
    """
    Recarrega o modelo campeão do disco.
    Deve ser chamado após o pipeline de retraining promover uma nova versão,
    para que a API passe a servir o modelo mais recente sem reiniciar o processo.
    """
    _load_model_into_state()
    if _state["metadata"] is None:
        raise HTTPException(status_code=404, detail="Nenhum modelo campeão encontrado após reload.")
    return ModelInfo(**_state["metadata"])


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    """Recebe os dados de um passageiro e retorna a previsão de sobrevivência."""
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Nenhum modelo treinado disponível. Execute o pipeline de treino primeiro.",
        )

    model = _state["model"]
    X = _build_feature_row(passenger)

    try:
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.error(f"Erro durante a inferência: {e}")
        raise HTTPException(status_code=500, detail=f"Erro durante a inferência: {e}")

    return PredictionOutput(
        survived=pred,
        survival_probability=round(proba, 4),
        model_used=_state["metadata"]["model_name"],
        model_version=_state["metadata"]["version"],
    )


@app.get("/")
def root():
    return {
        "message": "Titanic Survival Prediction API",
        "docs": "/docs",
        "health": "/health",
    }
