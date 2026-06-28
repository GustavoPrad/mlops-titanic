"""
app/streamlit_app.py
Interface simples de consumo da API de inferência.

Permite que o usuário preencha os dados de um passageiro do Titanic
e veja a previsão de sobrevivência feita pelo modelo campeão.

Para rodar:
    streamlit run app/streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.caption("Interface de consumo do pipeline de MLOps — modelo campeão em produção")

# ---------------------------------------------------------------
# Status do modelo em produção
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Status do Modelo")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        info = requests.get(f"{API_URL}/model/info", timeout=5).json()

        if health["model_loaded"]:
            st.success(f"API online — modelo carregado ✅")
            st.metric("Modelo campeão", info["model_name"])
            st.metric("Versão", info["version"])
            st.metric("F1-score", f"{info['metrics']['f1_score']:.3f}")
            st.metric("ROC-AUC", f"{info['metrics']['roc_auc']:.3f}")
            st.caption(f"Treinado em: {info['trained_at']}")
        else:
            st.error("Nenhum modelo carregado na API.")
    except requests.exceptions.ConnectionError:
        st.error(f"Não foi possível conectar à API em {API_URL}")

    if st.button("🔄 Recarregar modelo (após retraining)"):
        try:
            r = requests.post(f"{API_URL}/model/reload", timeout=5)
            if r.status_code == 200:
                st.success("Modelo recarregado com sucesso!")
                st.rerun()
            else:
                st.error(f"Erro: {r.json().get('detail')}")
        except requests.exceptions.ConnectionError:
            st.error("Não foi possível conectar à API.")

# ---------------------------------------------------------------
# Formulário de entrada
# ---------------------------------------------------------------
st.subheader("Dados do passageiro")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Classe da passagem", [1, 2, 3], index=2)
    sex = st.selectbox("Sexo", ["male", "female"])
    age = st.slider("Idade", 0, 90, 30)
    fare = st.number_input("Valor da passagem (£)", min_value=0.0, value=15.0, step=1.0)

with col2:
    sibsp = st.number_input("Nº de irmãos/cônjuges a bordo", min_value=0, max_value=10, value=0)
    parch = st.number_input("Nº de pais/filhos a bordo", min_value=0, max_value=10, value=0)
    embarked = st.selectbox(
        "Porto de embarque", ["S", "C", "Q"],
        format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x],
    )

st.divider()

if st.button("🔮 Prever sobrevivência", type="primary", use_container_width=True):
    payload = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            proba = result["survival_probability"]

            if result["survived"] == 1:
                st.success(f"✅ Sobreviveria — probabilidade de {proba:.1%}")
            else:
                st.error(f"❌ Não sobreviveria — probabilidade de sobrevivência de {proba:.1%}")

            st.progress(proba)
            st.caption(
                f"Previsão feita pelo modelo **{result['model_used']}** "
                f"(versão {result['model_version']})"
            )
        else:
            st.error(f"Erro na API: {response.json().get('detail', response.text)}")
    except requests.exceptions.ConnectionError:
        st.error(f"Não foi possível conectar à API em {API_URL}. Ela está rodando?")

st.divider()
st.caption(
    "Este app consome a API FastAPI (`api/main.py`), que serve sempre o último "
    "modelo campeão salvo pelo pipeline de MLOps (`src/pipeline.py`)."
)
