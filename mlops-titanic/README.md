# 🚢 Titanic MLOps — Pipeline Completo de Produtização de Machine Learning

Projeto end-to-end de **MLOps** que simula um fluxo real de produção de Machine
Learning: ingestão de dados, EDA, treino de múltiplos modelos, seleção
automática do melhor modelo, versionamento, deploy via API e ciclo
automatizado de retraining.

**Dataset:** [Titanic - Kaggle](https://www.kaggle.com/c/titanic) (espelho
público via `datasciencedojo/datasets`), problema de classificação binária:
prever se um passageiro sobreviveu (`Survived`: 0 ou 1).

---

## 🗺️ Visão geral da arquitetura

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  data/raw   │ --> │  src/        │ --> │  models/      │ --> │  api/        │
│  titanic.csv│     │  pipeline.py │     │  champion.pkl │     │  main.py     │
└─────────────┘     │  (treino +   │     │  + registry   │     │  (FastAPI)   │
                     │   seleção)   │     │   .json       │     └──────┬───────┘
                     └──────────────┘     └───────────────┘            │
                            ▲                                          ▼
                     ┌──────┴───────┐                          ┌──────────────┐
                     │ src/         │                          │  app/        │
                     │ auto_retrain │ <-- novos dados em       │  streamlit_  │
                     │ .py (watcher)│     data/incoming/       │  app.py      │
                     └──────────────┘                          └──────────────┘
```

## 📁 Estrutura do projeto

```
mlops-titanic/
├── config/
│   └── config.yaml          # todos os parâmetros do pipeline (modelos, métricas, split)
├── data/
│   ├── raw/titanic.csv       # dataset original (versionado)
│   ├── processed/            # dados limpos, gerados pelo pipeline
│   └── incoming/             # "caixa de entrada" de novos dados (ciclo automatizado)
├── models/
│   ├── champion_model.pkl        # modelo vencedor atual (artefato de produção)
│   ├── champion_metadata.json    # métricas/versão do campeão atual
│   └── model_registry.json       # histórico completo de versões treinadas
├── reports/figures/          # gráficos gerados pela EDA
├── src/
│   ├── utils.py               # config loader + logging
│   ├── data_processing.py     # ingestão, limpeza, feature engineering
│   ├── eda.py                 # análise exploratória multivariada
│   ├── train.py                # treino dos 3 modelos
│   ├── evaluate.py             # avaliação com múltiplas métricas
│   ├── model_selection.py      # seleção automática + critério de desempate
│   ├── persistence.py          # salvar/carregar modelo + registry
│   ├── pipeline.py              # ORQUESTRADOR — junta todas as etapas
│   └── auto_retrain.py          # ciclo automatizado (watcher de novos dados)
├── api/
│   ├── main.py                  # API FastAPI de inferência
│   └── schemas.py                # contratos de entrada/saída (Pydantic)
├── app/
│   └── streamlit_app.py          # interface simples de consumo
├── tests/                         # testes automatizados (pytest)
├── .github/workflows/              # CI/CD (GitHub Actions)
├── Dockerfile / Dockerfile.app / docker-compose.yml
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🚀 Como executar

### 1. Instalação

```bash
git clone <seu-repositorio>
cd mlops-titanic
python -m venv venv && source venv/bin/activate   # opcional, recomendado
pip install -r requirements.txt
```

### 2. Análise exploratória (EDA)

```bash
python -m src.eda
```
Gera estatísticas descritivas, matriz de correlação e uma análise
multivariada (taxa de sobrevivência por **Sexo × Classe**), além de salvar
gráficos em `reports/figures/`.

### 3. Rodar o pipeline completo (treino + seleção + persistência)

```bash
python -m src.pipeline
```

O que acontece, em ordem:
1. **Ingestão** do CSV bruto (`data/raw/titanic.csv`)
2. **Limpeza** (imputação de `Age`/`Fare`/`Embarked`) e **feature engineering**
   (`FamilySize = SibSp + Parch + 1`)
3. **Treino de 3 modelos**: Logistic Regression, Random Forest e XGBoost
   (todos embrulhados em um `sklearn.Pipeline` com pré-processamento —
   `StandardScaler` + `OneHotEncoder`)
4. **Avaliação** com 5 métricas: accuracy, F1-score, ROC-AUC, precision, recall
5. **Seleção automática** do melhor modelo por F1-score (critério de
   desempate: ROC-AUC)
6. **Decisão de promoção**: o modelo só substitui o campeão atual se realmente
   melhorar a métrica principal (evita troca por ruído estatístico)
7. **Persistência**: salva `.pkl` do modelo + metadados + registro de versão

Saída esperada (resumida):
```
=== COMPARATIVO DE MODELOS ===
                     accuracy  f1_score  roc_auc  precision  recall
random_forest          0.8101    0.7302   0.8444     0.8070  0.6667
logistic_regression    0.8045    0.7244   0.8435     0.7931  0.6667
xgboost                0.7989    0.7143   0.8231     0.7895  0.6522

Melhor modelo do batch: random_forest
Promovido a campeão:    True
Versão registrada:      1
```

### 4. Subir a API de inferência

```bash
uvicorn api.main:app --reload --port 8000
```

Documentação interativa (Swagger): **http://localhost:8000/docs**

**Endpoints:**

| Método | Rota            | Descrição                                            |
|--------|-----------------|-------------------------------------------------------|
| GET    | `/health`       | Healthcheck                                           |
| GET    | `/model/info`   | Metadados do modelo campeão em produção               |
| POST   | `/predict`      | Recebe dados de um passageiro, retorna a previsão     |
| POST   | `/model/reload` | Recarrega o modelo do disco (após retraining)         |

Exemplo de chamada:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass":1,"Sex":"female","Age":29,"SibSp":0,"Parch":0,"Fare":80.0,"Embarked":"C"}'
```
```json
{"survived":1,"survival_probability":0.9943,"model_used":"random_forest","model_version":1}
```

### 5. Interface de consumo (Streamlit)

```bash
streamlit run app/streamlit_app.py
```
Abre em **http://localhost:8501** — formulário simples para simular um
passageiro e ver a previsão em tempo real, consumindo a API.

> A API precisa estar rodando antes de abrir o app (passo 4).

### 6. Ciclo automatizado de MLOps (retraining contínuo)

Simula o requisito de "quando novos dados estiverem disponíveis, o pipeline
roda automaticamente":

```bash
# Modo contínuo: observa data/incoming/ e dispara o pipeline a cada novo CSV
python -m src.auto_retrain --watch

# Modo único: processa arquivos pendentes uma vez (ideal para cron/Airflow/CI)
python -m src.auto_retrain --once
```

Basta colocar um novo arquivo `.csv` (mesmas colunas do Titanic) em
`data/incoming/` que o sistema automaticamente:
1. Mescla os novos dados à base histórica
2. Re-executa o pipeline completo (treino dos 3 modelos)
3. Compara o novo melhor modelo com o campeão atual
4. **Promove** a nova versão apenas se houver melhoria real
5. Atualiza `model_registry.json` com o histórico completo
6. Arquiva o CSV processado em `data/incoming/_processed/`

Depois, chame `POST /model/reload` (ou clique em "🔄 Recarregar modelo" no
Streamlit) para a API passar a servir a nova versão sem reiniciar o processo.

---

## 🐳 Deploy com Docker

```bash
docker compose up --build
```
- API: http://localhost:8000/docs
- App: http://localhost:8501

O `docker-compose.yml` monta `models/` e `data/` como volumes, então rodar
`python -m src.pipeline` (ou `auto_retrain`) fora do container já reflete na
API após um `/model/reload`.

---

## ⚙️ Configuração (`config/config.yaml`)

Todos os parâmetros relevantes do pipeline ficam centralizados em YAML —
**nenhum hiperparâmetro fica hardcoded no código**:

```yaml
evaluation:
  primary_metric: "f1_score"      # métrica usada para escolher o campeão
  tiebreaker_metric: "roc_auc"     # critério de desempate
  improvement_threshold: 0.0       # margem mínima para promover um novo modelo
```

Quer testar outro modelo principal de seleção? Troque `primary_metric` para
`roc_auc` ou `accuracy` e rode `python -m src.pipeline` novamente — nenhuma
linha de código precisa mudar.

---

## 🧪 Testes automatizados

```bash
pytest tests/ -v
```
Cobre: pipeline de dados, treino dos 3 modelos, métricas de avaliação,
lógica de seleção/desempate, lógica de promoção e os endpoints da API.

---

## 🔄 CI/CD (GitHub Actions)

`.github/workflows/mlops_pipeline.yml` define 3 jobs:
1. **test** — roda a suíte pytest a cada push/PR
2. **retrain** — executa o pipeline completo e publica o modelo como artefato
   do workflow (simula retraining automatizado via CI)
3. **build-and-push** — builda a imagem Docker da API (push para um registry
   é opcional, requer configurar secrets)

Pode ser disparado manualmente (`workflow_dispatch`) para simular a chegada
de "novos dados em produção".

---

## 📦 Publicando o projeto

### GitHub
```bash
git init
git add .
git commit -m "feat: pipeline completo de MLOps - Titanic"
git branch -M main
git remote add origin <url-do-seu-repositorio>
git push -u origin main
```

### Deploy da API (opções gratuitas/simples)
- **Render** / **Railway**: conectam direto no repositório GitHub e detectam
  o `Dockerfile` automaticamente. Configure a porta `8000` e pronto.
- **Fly.io**: `fly launch` detecta o Dockerfile e gera o `fly.toml`.
- Qualquer VM com Docker: `docker compose up -d --build`.

> Lembre-se de versionar `models/champion_model.pkl` (ou gerar via CI antes do
> deploy) para a API ter um modelo carregado em produção desde o primeiro boot.

---

## 🧠 Decisões de design (e por quê)

- **Pipeline do sklearn** (pré-processamento + modelo no mesmo objeto): evita
  inconsistência entre o que foi usado no treino e o que é usado na inferência
  — o `.pkl` salvo já contém todo o fluxo de transformação.
- **F1-score como métrica principal**: o Titanic tem classes desbalanceadas
  (~38% sobreviventes), então accuracy isolada é enganosa; F1 equilibra
  precisão e recall. ROC-AUC funciona como desempate por avaliar a separação
  de classes de forma mais robusta a thresholds.
- **`improvement_threshold` configurável**: protege contra a troca de modelo
  em produção apenas por flutuação estatística entre execuções.
- **Escrita atômica do `model_registry.json`** (tmp + replace): evita
  corromper o histórico se o processo for interrompido no meio da escrita.
- **API com estado em memória + endpoint `/model/reload`**: simula o padrão
  real de produção onde o serviço de inferência não precisa reiniciar para
  servir uma nova versão do modelo.
