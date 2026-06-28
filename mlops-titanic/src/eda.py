"""
src/eda.py
Análise exploratória simples (multivariada básica) do dataset Titanic.
Gera:
- Estatísticas descritivas
- Matriz de correlação entre variáveis numéricas
- Taxa de sobrevivência cruzada por Sexo x Classe (análise multivariada)
- Gráficos salvos em reports/

Pode ser executado isoladamente: python -m src.eda
"""

import matplotlib

matplotlib.use("Agg")  # backend não interativo (ambiente sem display)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_processing import run_data_pipeline
from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas das variáveis numéricas."""
    stats = df.describe()
    logger.info("Estatísticas descritivas calculadas")
    return stats


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlação entre variáveis numéricas (análise multivariada)."""
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()
    return corr


def survival_by_sex_and_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Análise multivariada: taxa de sobrevivência cruzando Sexo x Classe.
    Mostra como duas variáveis combinadas afetam o target.
    """
    pivot = df.pivot_table(
        values="Survived", index="Sex", columns="Pclass", aggfunc="mean"
    )
    return pivot


def generate_plots(df: pd.DataFrame, corr: pd.DataFrame, output_dir):
    """Gera e salva gráficos básicos de EDA."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap de correlação
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlação - Titanic")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=120)
    plt.close()

    # Sobrevivência por classe e sexo
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex")
    plt.title("Taxa de Sobrevivência por Classe e Sexo")
    plt.ylabel("Taxa de Sobrevivência")
    plt.tight_layout()
    plt.savefig(output_dir / "survival_by_class_sex.png", dpi=120)
    plt.close()

    # Distribuição de idade por sobrevivência
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30, multiple="stack")
    plt.title("Distribuição de Idade por Sobrevivência")
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution.png", dpi=120)
    plt.close()

    logger.info(f"Gráficos de EDA salvos em {output_dir}")


def run_eda(config: dict = None):
    """Executa a análise exploratória completa e imprime um resumo no console."""
    if config is None:
        config = load_config()

    df = run_data_pipeline(config)

    stats = descriptive_stats(df)
    corr = correlation_matrix(df)
    pivot = survival_by_sex_and_class(df)

    print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
    print(stats)

    print("\n=== MATRIZ DE CORRELAÇÃO ===")
    print(corr.round(2))

    print("\n=== TAXA DE SOBREVIVÊNCIA: SEXO x CLASSE (multivariada) ===")
    print(pivot.round(2))

    output_dir = resolve_path("reports/figures")
    generate_plots(df, corr, output_dir)

    return {"stats": stats, "correlation": corr, "survival_pivot": pivot}


if __name__ == "__main__":
    run_eda()
