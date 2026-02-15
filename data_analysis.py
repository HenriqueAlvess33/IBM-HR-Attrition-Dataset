import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# ===================== CONFIGURAÇÕES INICIAIS =====================
st.set_page_config(
    page_title="Análise de Turnover - IBM HR Attrition",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="./images/ibm_logo.png",  # Certifique-se de que o arquivo existe
)

# ===================== VARIÁVEIS DE SESSÃO =====================
session_defaults = {
    "data": None,
    "target_col": "Attrition",  # coluna alvo padrão
    "categorical_cols": [],  # lista de colunas categóricas do dataset
    "numerical_cols": [],  # lista de colunas numéricas do dataset
    "selected_categorical": [],  # seleção do usuário
    "selected_numerical": [],  # seleção do usuário
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ===================== FUNÇÕES AUXILIARES =====================
@st.cache_data
def load_data(uploaded_file):
    """Carrega um arquivo CSV para um DataFrame."""
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Não foi possível carregar o arquivo: {e}")
        return None


def infer_column_types(df, target_col):
    """
    Infere colunas categóricas e numéricas, excluindo a coluna alvo.
    Retorna duas listas.
    """
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target de ambas (se estiver presente)
    if target_col in categorical:
        categorical.remove(target_col)
    if target_col in numerical:
        numerical.remove(target_col)

    return categorical, numerical


def convert_target_to_binary(df, target_col, positive_value):
    """
    Converte a coluna alvo para 0/1 com base no valor positivo escolhido.
    A coluna é modificada inplace.
    """
    df[target_col] = df[target_col].apply(lambda x: 1 if x == positive_value else 0)
    return df


def calculate_attrition_proportions(df, group_col, target_col):
    """
    Calcula contagens e proporções de attrition para uma coluna categórica.
    Retorna um DataFrame com colunas: group_col, target_col, Contagem, Total, Proporcao.
    """
    counts = df.groupby([group_col, target_col]).size().reset_index(name="Contagem")
    totals = df.groupby(group_col).size().reset_index(name="Total")
    merged = counts.merge(totals, on=group_col)
    merged["Proporcao"] = merged["Contagem"] / merged["Total"]
    return merged


def plot_attrition_proportions(
    proportions_df, group_col, target_col, palette="viridis", figsize=(10, 6)
):
    """Gráfico de barras da proporção de attrition por categoria."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=proportions_df,
        x=group_col,
        y="Proporcao",
        hue=target_col,
        palette=palette,
        ax=ax,
    )
    ax.set_title(
        f"Proporção de '{target_col}' por '{group_col}'", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel(f"Categorias de '{group_col}'", fontsize=12)
    ax.set_ylabel("Proporção", fontsize=12)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)

    # Rótulos percentuais
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{height:.1%}",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )
    fig.tight_layout()
    return fig


def plot_normalized_distribution(df, continuous_var, target_col, bins=30):
    """
    Plota distribuição normalizada (KDE e histograma de proporções) de uma variável contínua.
    Retorna a figura matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dados separados
    stay = df[df[target_col] == 0][continuous_var].dropna()
    leave = df[df[target_col] == 1][continuous_var].dropna()

    # Estatísticas
    stay_mean = stay.mean()
    leave_mean = leave.mean()

    # 1. KDE normalizado
    axes[0].set_title(
        f"Densidade Normalizada: {continuous_var} por Turnover",
        fontsize=14,
        fontweight="bold",
    )
    sns.kdeplot(
        data=stay,
        ax=axes[0],
        label="Ficam (0)",
        fill=True,
        alpha=0.5,
        color="blue",
        linewidth=2,
    )
    sns.kdeplot(
        data=leave,
        ax=axes[0],
        label="Saem (1)",
        fill=True,
        alpha=0.5,
        color="red",
        linewidth=2,
    )
    axes[0].axvline(
        stay_mean,
        color="blue",
        linestyle="--",
        alpha=0.8,
        label=f"Média Ficam: {stay_mean:.1f}",
    )
    axes[0].axvline(
        leave_mean,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Média Saem: {leave_mean:.1f}",
    )
    axes[0].set_xlabel(continuous_var, fontsize=12)
    axes[0].set_ylabel("Densidade de Probabilidade", fontsize=12)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # 2. Histograma de proporções (stacked)
    axes[1].set_title(
        f"Proporção por Faixa: {continuous_var}", fontsize=14, fontweight="bold"
    )
    min_val = df[continuous_var].min()
    max_val = df[continuous_var].max()
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    proportions = []
    bin_centers = []
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        in_bin = df[(df[continuous_var] >= lower) & (df[continuous_var] < upper)]
        if len(in_bin) > 0:
            total = len(in_bin)
            leave_prop = len(in_bin[in_bin[target_col] == 1]) / total
            stay_prop = 1 - leave_prop
            proportions.append((stay_prop, leave_prop))
            bin_centers.append((lower + upper) / 2)

    proportions = np.array(proportions)
    bin_centers = np.array(bin_centers)

    axes[1].bar(
        bin_centers,
        proportions[:, 0],
        width=(max_val - min_val) / bins * 0.8,
        color="blue",
        alpha=0.7,
        label="Ficam (0)",
    )
    axes[1].bar(
        bin_centers,
        proportions[:, 1],
        bottom=proportions[:, 0],
        width=(max_val - min_val) / bins * 0.8,
        color="red",
        alpha=0.7,
        label="Saem (1)",
    )

    overall_leave_rate = len(df[df[target_col] == 1]) / len(df)
    axes[1].axhline(
        y=overall_leave_rate,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Taxa Geral: {overall_leave_rate:.1%}",
    )
    axes[1].set_xlabel(continuous_var, fontsize=12)
    axes[1].set_ylabel("Proporção no Bin", fontsize=12)
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Estatísticas adicionais como texto
    stats_text = (
        f"Estatísticas - {continuous_var}:\n"
        f"• Total Ficam: {len(stay):,} ({len(stay)/len(df):.1%})\n"
        f"• Total Saem: {len(leave):,} ({len(leave)/len(df):.1%})\n"
        f"• Média Ficam: {stay_mean:.1f}\n"
        f"• Média Saem: {leave_mean:.1f}\n"
        f"• Diferença: {abs(stay_mean - leave_mean):.1f}"
    )
    fig.text(
        0.02,
        0.02,
        stats_text,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    return fig


def plot_absolute_distributions(df, numerical_vars, target_col):
    """
    Cria uma grade de subplots para variáveis numéricas (modo absoluto).
    Para variáveis com poucos valores únicos (<10), usa gráfico de barras de proporção.
    Caso contrário, usa violinplot.
    """
    n_cols = 4
    n_vars = len(numerical_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

    # Achatar axes se necessário
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, var in enumerate(numerical_vars):
        ax = axes[idx]
        if df[var].nunique() < 10:
            # Tratar como categórica ordinal
            props = calculate_attrition_proportions(df, var, target_col)
            sns.barplot(
                data=props,
                x=var,
                y="Proporcao",
                hue=target_col,
                ax=ax,
                palette="viridis",
            )
            ax.set_title(f"Proporção por {var}", fontsize=12)
            ax.set_xlabel(var)
            ax.set_ylabel("Proporção")
            ax.tick_params(axis="x", rotation=45)
            # Rótulos percentuais
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + 0.02,
                            f"{height:.1%}",
                            ha="center",
                            fontsize=8,
                            fontweight="bold",
                        )
        else:
            sns.violinplot(
                x=target_col, y=var, data=df, inner="box", ax=ax, palette="Set2"
            )
            ax.set_title(f"Distribuição de {var}", fontsize=12)
            ax.set_xlabel(f"{target_col} (0 = Não, 1 = Sim)")
            ax.set_ylabel(var)

    # Remove eixos não utilizados
    for j in range(len(numerical_vars), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


# ===================== LAYOUT PRINCIPAL =====================
def main():
    # Cabeçalho com imagem
    try:
        img = Image.open("./images/office.jpg")
        # Redimensionar mantendo proporção
        max_height = 500
        if img.height > max_height:
            new_height = max_height
            new_width = int(img.width * (new_height / img.height))
            img = img.resize((new_width, new_height))
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, use_container_width=False)
    except FileNotFoundError:
        pass  # imagem opcional

    # Barra lateral
    with st.sidebar:
        st.header("1. Carregar dados")
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type=["csv"],
            help="Dataset com coluna alvo de turnover (ex: Attrition)",
        )

        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.data = df
                st.success("Dados carregados com sucesso!")
                st.write(f"**Dimensões:** {df.shape[0]} linhas, {df.shape[1]} colunas")
                st.write("**Primeiras linhas:**")
                st.dataframe(df.head())

                # Permitir selecionar a coluna alvo
                all_cols = df.columns.tolist()
                target = st.selectbox(
                    "Selecione a coluna alvo (turnover)",
                    options=all_cols,
                    index=(
                        all_cols.index(st.session_state.target_col)
                        if st.session_state.target_col in all_cols
                        else 0
                    ),
                )
                st.session_state.target_col = target

                # Converter a coluna alvo para binária (0/1) se necessário
                target_series = df[target]
                if (
                    target_series.dtype == "object"
                    or str(target_series.dtype) == "category"
                ):
                    unique_vals = target_series.unique()
                    if len(unique_vals) == 2:
                        # Perguntar ao usuário qual valor representa turnover (1)
                        positive_val = st.selectbox(
                            f"Qual valor em '{target}' indica turnover?",
                            options=unique_vals,
                        )
                        # Mapear: positivo -> 1, outro -> 0
                        df[target] = df[target].apply(
                            lambda x: 1 if x == positive_val else 0
                        )
                        st.info(
                            f"Coluna '{target}' convertida: {positive_val} → 1, outro → 0"
                        )
                    else:
                        st.error(
                            f"A coluna alvo deve ter exatamente 2 valores únicos. Encontrados: {unique_vals}"
                        )
                        st.stop()
                else:
                    # Se já é numérica, verificar se contém apenas 0 e 1
                    unique_vals = target_series.unique()
                    if set(unique_vals) not in ({0, 1}, {0}, {1}):
                        st.warning(
                            f"A coluna alvo numérica contém valores diferentes de 0 e 1: {unique_vals}. Certifique-se de que 1 indica turnover."
                        )

                # Atualizar o dataframe na sessão com a coluna convertida
                st.session_state.data = df

                # Inferir tipos
                cat_cols, num_cols = infer_column_types(df, target)
                st.session_state.categorical_cols = cat_cols
                st.session_state.numerical_cols = num_cols

                st.markdown("---")
                st.header("2. Selecionar variáveis")
                st.session_state.selected_categorical = st.multiselect(
                    "Variáveis categóricas para análise", options=cat_cols, default=[]
                )
                st.session_state.selected_numerical = st.multiselect(
                    "Variáveis numéricas para análise", options=num_cols, default=[]
                )
            else:
                st.session_state.data = None
        else:
            st.info("Carregue o dataset para iniciar.")
            st.session_state.data = None

    # Título principal
    st.title("Análise de Turnover - IBM HR Employee Attrition")

    # Verificar se dados foram carregados
    if st.session_state.data is None:
        st.warning("Por favor, carregue um arquivo CSV na barra lateral.")
        return

    # Abas
    tab1, tab2 = st.tabs(["📊 Variáveis Categóricas", "📈 Variáveis Numéricas"])

    # ===================== ABA 1: CATEGÓRICAS =====================
    with tab1:
        if not st.session_state.selected_categorical:
            st.info("Selecione pelo menos uma variável categórica na barra lateral.")
        else:
            for i, cat in enumerate(st.session_state.selected_categorical):
                st.subheader(f"Análise: {cat}")
                if cat not in st.session_state.data.columns:
                    st.warning(f"Coluna '{cat}' não encontrada no dataset.")
                    continue
                props = calculate_attrition_proportions(
                    st.session_state.data, cat, st.session_state.target_col
                )
                fig = plot_attrition_proportions(
                    props, cat, st.session_state.target_col
                )
                st.pyplot(fig)
                if i < len(st.session_state.selected_categorical) - 1:
                    st.markdown("---")

    # ===================== ABA 2: NUMÉRICAS =====================
    with tab2:
        if not st.session_state.selected_numerical:
            st.info("Selecione pelo menos uma variável numérica na barra lateral.")
        else:
            viz_mode = st.radio(
                "Tipo de visualização",
                ["Gráficos absolutos", "Gráficos normalizados (KDE + proporções)"],
                horizontal=True,
            )

            if viz_mode == "Gráficos absolutos":
                fig = plot_absolute_distributions(
                    st.session_state.data,
                    st.session_state.selected_numerical,
                    st.session_state.target_col,
                )
                st.pyplot(fig)
            else:
                overall_rate = st.session_state.data[st.session_state.target_col].mean()
                st.write(f"**Taxa geral de turnover:** {overall_rate:.1%}")

                for var in st.session_state.selected_numerical:
                    st.subheader(f"Análise: {var}")
                    fig = plot_normalized_distribution(
                        st.session_state.data, var, st.session_state.target_col, bins=15
                    )
                    st.pyplot(fig)

                    # Análise de quartis
                    st.markdown("##### Risco por quartil")
                    # Criar cópia temporária para não modificar original
                    temp_df = st.session_state.data.copy()
                    try:
                        temp_df["quartil"] = pd.qcut(
                            temp_df[var],
                            4,
                            labels=["Q1 (Baixo)", "Q2", "Q3", "Q4 (Alto)"],
                        )
                    except ValueError:
                        # Fallback para dados com muitos valores repetidos
                        quartiles = temp_df[var].quantile([0.25, 0.5, 0.75])
                        bins = [-float("inf")] + quartiles.tolist() + [float("inf")]
                        labels = ["Q1 (Baixo)", "Q2", "Q3", "Q4 (Alto)"]
                        temp_df["quartil"] = pd.cut(
                            temp_df[var], bins=bins, labels=labels
                        )

                    risk = (
                        temp_df.groupby("quartil")[st.session_state.target_col]
                        .mean()
                        .reset_index()
                    )
                    risk.columns = ["Quartil", "Taxa de Turnover"]
                    risk["Risco Relativo"] = risk["Taxa de Turnover"] / overall_rate

                    # Formatar para exibição
                    st.dataframe(
                        risk.style.format(
                            {"Taxa de Turnover": "{:.1%}", "Risco Relativo": "{:.2f}x"}
                        )
                    )
                    st.markdown("---")


# ===================== EXECUÇÃO =====================
if __name__ == "__main__":
    main()
