"""Wrapper do RandomForestClassifier do scikit-learn."""
from math import sqrt

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def RandomForest(
    n_features: int,
    X_train: pd.DataFrame,
    labels: pd.Series,
    n_estimators: int,
) -> RandomForestClassifier:
    """Treina um RandomForestClassifier.

    O número de features consideradas por divisão usa a heurística
    ``max_features = sqrt(n_features)``, comum para problemas de classificação.

    Args:
        n_features: Total de features do dataset (usado para calcular ``max_features``).
        X_train: Features de treinamento.
        labels: Labels de treinamento.
        n_estimators: Número de árvores na floresta.

    Returns:
        Modelo treinado.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=int(sqrt(n_features)),
        random_state=42,
    )
    model.fit(X_train, labels)
    return model


def get_metrics(
    model: RandomForestClassifier,
    validation: pd.DataFrame,
    labels: pd.Series,
) -> tuple[float, float, float, float]:
    """Calcula métricas de classificação no split de validação.

    Args:
        model: Modelo treinado.
        validation: Features de validação.
        labels: Labels reais de validação.

    Returns:
        Tupla ``(accuracy, f1, precision, recall)``.
    """
    preds = model.predict(validation)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    recall = recall_score(labels, preds)
    return accuracy, f1, precision, recall
