"""Pré-processamento de dados para o Chameleon-IDS.

Suporta os datasets NSL-KDD e CICIDS2017.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Nomes das colunas alvo e com valores infinitos no CICIDS2017
_CICIDS_LABEL_COL = "Label"
_CICIDS_BENIGN_LABEL = "BENIGN"
_CICIDS_INF_COLS = ("Flow Packets/s", "Flow Bytes/s")


def normalize_data(subset: pd.DataFrame) -> pd.DataFrame:
    """Normaliza as colunas numéricas com z-score (média 0, desvio 1).

    Args:
        subset: DataFrame com colunas numéricas a normalizar.

    Returns:
        DataFrame com as mesmas colunas numéricas normalizadas.
    """
    std_scaler = StandardScaler()
    colunas_numericas = subset.select_dtypes(include=["number"])
    normalizado = pd.DataFrame(
        std_scaler.fit_transform(colunas_numericas),
        columns=colunas_numericas.columns,
    )
    return normalizado


def split_train_test(
    df: pd.DataFrame,
    column_names: list[str],
    y: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Divide o dataset em treino e validação.

    Args:
        df: Base de dados completa.
        column_names: Lista de colunas de features.
        y: Labels do dataset.
        test_size: Fração destinada à validação.

    Returns:
        Tupla ``(x_train, y_train, x_val, y_val)``.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        df[column_names], y, test_size=test_size, random_state=42, stratify=y
    )
    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    return x_train, y_train, x_val, y_val


def split_train_val_test(
    df: pd.DataFrame,
    column_names: list[str],
    y: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Divide o dataset em treino, validação e teste.

    O segmento não-treino é dividido igualmente entre validação e teste.

    Args:
        df: Base de dados completa.
        column_names: Lista de colunas de features.
        y: Labels do dataset.
        test_size: Fração total destinada a validação + teste.

    Returns:
        Tupla ``(x_train, y_train, x_val, y_val, x_test, y_test)``.
    """
    x_train, x_val_test, y_train, y_val_test = train_test_split(
        df[column_names], y, test_size=test_size, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test
    )
    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_nslkdd(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Pré-processa o dataset NSL-KDD.

    Etapas: strip de nomes de colunas, remoção de NaN, encoding de colunas
    categóricas, normalização z-score e extração de labels.

    Args:
        df: DataFrame bruto do NSL-KDD (com coluna ``class``).

    Returns:
        Tupla ``(df_processado, column_names, y)`` onde ``y`` é binário
        (0 = normal, 1 = ataque).
    """
    df.columns = [col.strip().replace("'", "") for col in df.columns]

    df = df.dropna()

    column_names: list[str] = df.drop(labels="class", axis=1).columns.tolist()
    y: pd.Series = df["class"].apply(lambda c: 0 if c == "normal" else 1)

    df_not_numeric = df.select_dtypes(exclude=[np.number])
    encoder = LabelEncoder()
    for column in df_not_numeric.columns:
        df[column] = encoder.fit_transform(df[column])

    df = normalize_data(df)
    df = df.drop(labels="class", axis=1)

    return df, column_names, y


# Alias legado
preprocessing = preprocess_nslkdd


def preprocess_cicids(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Pré-processa o dataset CICIDS2017.

    Etapas: strip de nomes de colunas, remoção de duplicatas e NaN,
    substituição de valores infinitos, encoding, normalização e extração
    de labels.

    Args:
        df: DataFrame bruto do CICIDS2017 (com coluna ``Label``).

    Returns:
        Tupla ``(df_processado, column_names, y)`` onde ``y`` é binário
        (0 = BENIGN, 1 = ataque).
    """
    df.columns = [col.strip() for col in df.columns]

    initial_len = df.shape[0]
    df = df.drop_duplicates()
    print(
        f"Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} "
        f"| Descartadas {initial_len - df.shape[0]} duplicadas"
    )

    initial_len = df.shape[0]
    df = df.dropna()
    print(
        f"Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} "
        f"| Descartados {initial_len - df.shape[0]} registros com valores NA"
    )

    df = df.reset_index(drop=True)

    # Substituir infinitos nas colunas de taxa por valor máximo finito
    for col in _CICIDS_INF_COLS:
        if col in df.columns:
            max_finite = df[np.isfinite(df[col])][col].max()
            df.loc[df[col] == np.inf, col] = max_finite

    y: pd.Series = df[_CICIDS_LABEL_COL].apply(lambda c: 0 if c == _CICIDS_BENIGN_LABEL else 1)

    df_not_numeric = df.select_dtypes(exclude=[np.number])
    encoder = LabelEncoder()
    for column in df_not_numeric.columns:
        df[column] = encoder.fit_transform(df[column])

    column_names: list[str] = df.drop(labels=_CICIDS_LABEL_COL, axis=1).columns.tolist()
    df = df.drop(labels=_CICIDS_LABEL_COL, axis=1)
    df = normalize_data(df)

    return df, column_names, y


# Alias legado
preprocessing_CICS = preprocess_cicids


def particle_choices(pos: list, column_names: list[str], n_features: int) -> list[str]:
    """Retorna os nomes das colunas selecionadas pela partícula.

    Args:
        pos: Posição da partícula (``pos[1..n_features]`` são as flags de seleção).
        column_names: Nomes de todas as colunas disponíveis.
        n_features: Número total de features.

    Returns:
        Lista de nomes das colunas com flag 1.
    """
    return [column_names[i - 1] for i in range(1, n_features + 1) if pos[i] == 1]


def get_optimal_subsets(
    df: pd.DataFrame,
    optimal_solution: list,
    column_names: list[str],
    y: pd.Series,
    test_size: float,
    n_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Obtém as partições treino/val/teste com as features da solução ótima do PSO.

    Args:
        df: Base de dados completa.
        optimal_solution: Posição da melhor partícula global.
        column_names: Nomes de todas as colunas disponíveis.
        y: Labels do dataset.
        test_size: Fração para validação + teste.
        n_features: Número total de features.

    Returns:
        Tupla ``(x_train, x_val, x_test, y_train, y_val, y_test)``
        com apenas as features selecionadas.
    """
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(
        df, column_names, y, test_size
    )
    chosen_columns = particle_choices(optimal_solution, column_names, n_features)
    return (
        x_train[chosen_columns],
        x_val[chosen_columns],
        x_test[chosen_columns],
        y_train,
        y_val,
        y_test,
    )



def transform_min_max_scaler(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    benign_x_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aplica escalonamento Min-Max nos quatro splits de dados.

    O scaler é ajustado apenas no ``x_train`` e aplicado nos demais.

    Args:
        x_train: Dados de treinamento.
        x_val: Dados de validação.
        x_test: Dados de teste.
        benign_x_train: Partição benigna para treinar o Autoencoder.

    Returns:
        Tupla ``(x_train, x_val, x_test, benign_x_train)`` escalonados.
    """
    scaler = MinMaxScaler().fit(x_train)
    return (
        scaler.transform(x_train),
        scaler.transform(x_val),
        scaler.transform(x_test),
        scaler.transform(benign_x_train),
    )


# Alias legado
transform_MinMaxScaler = transform_min_max_scaler


def get_time(start_time: float, end_time: float) -> None:
    """Imprime o tempo de execução formatado em horas, minutos e segundos.

    Args:
        start_time: Timestamp de início (``time.time()``).
        end_time: Timestamp de fim (``time.time()``).
    """
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = int(elapsed % 60)
    print(f"Tempo de execução: {hours} horas, {mins} minutos e {secs} segundos")
