"""
tests/test_api.py
Testes automatizados da API de inferência (FastAPI TestClient).
Pressupõe que o pipeline já foi executado ao menos uma vez
(existe um modelo campeão salvo em models/).

Rodar com: pytest tests/ -v
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_model_info_endpoint():
    response = client.get("/model/info")
    # 200 se já houver modelo treinado; 404 se o pipeline nunca rodou
    assert response.status_code in (200, 404)


def test_predict_endpoint_valid_payload():
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 29,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0,
        "Embarked": "C",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 503)  # 503 se nenhum modelo estiver carregado

    if response.status_code == 200:
        body = response.json()
        assert body["survived"] in (0, 1)
        assert 0.0 <= body["survival_probability"] <= 1.0
        assert "model_used" in body


def test_predict_endpoint_invalid_payload():
    payload = {"Pclass": 1, "Sex": "female"}  # campos obrigatórios faltando
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # erro de validação do Pydantic


def test_predict_endpoint_invalid_pclass():
    payload = {
        "Pclass": 9,  # fora do intervalo permitido (1-3)
        "Sex": "female",
        "Age": 29,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0,
        "Embarked": "C",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
