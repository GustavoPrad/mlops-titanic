"""
api/schemas.py
Schemas Pydantic usados pela API de inferência:
- PassengerInput: dados de um passageiro para previsão
- PredictionOutput: resultado retornado pela API
- ModelInfo: metadados do modelo campeão atualmente em produção
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PassengerInput(BaseModel):
    """
    Dados de entrada de um passageiro do Titanic, no mesmo formato
    do dataset original (antes do pré-processamento interno).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Pclass": 1,
                "Sex": "female",
                "Age": 29,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 80.0,
                "Embarked": "C",
            }
        }
    )

    Pclass: int = Field(..., ge=1, le=3, description="Classe da passagem (1, 2 ou 3)")
    Sex: str = Field(..., description="Sexo do passageiro: 'male' ou 'female'")
    Age: float = Field(..., ge=0, le=100, description="Idade do passageiro")
    SibSp: int = Field(..., ge=0, description="Nº de irmãos/cônjuges a bordo")
    Parch: int = Field(..., ge=0, description="Nº de pais/filhos a bordo")
    Fare: float = Field(..., ge=0, description="Valor pago pela passagem")
    Embarked: str = Field(..., description="Porto de embarque: 'C', 'Q' ou 'S'")


class PredictionOutput(BaseModel):
    survived: int = Field(..., description="0 = não sobreviveu, 1 = sobreviveu")
    survival_probability: float = Field(..., description="Probabilidade de sobrevivência (0 a 1)")
    model_used: str = Field(..., description="Nome do modelo campeão usado na previsão")
    model_version: int = Field(..., description="Versão do modelo campeão registrada")


class ModelInfo(BaseModel):
    model_name: str
    version: int
    metrics: dict
    trained_at: str
    promoted: bool


class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[int] = None
