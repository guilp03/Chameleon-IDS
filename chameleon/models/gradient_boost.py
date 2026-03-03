"""Wrapper do GradientBoostingClassifier do scikit-learn.

Este módulo usa exclusivamente ``sklearn.ensemble.GradientBoostingClassifier``
— não a biblioteca ``xgboost``.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def GradientBoost(
    X_train: pd.DataFrame,
    labels: pd.Series,
    n_estimators: int,
    learning_rate: float,
) -> GradientBoostingClassifier:
    """Treina um GradientBoostingClassifier.

    Args:
        X_train: Features de treinamento.
        labels: Labels de treinamento.
        n_estimators: Número de árvores.
        learning_rate: Taxa de aprendizado (shrinkage).

    Returns:
        Modelo treinado.
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42,
    )
    model.fit(X_train, labels)
    return model


def get_metrics(
    model: GradientBoostingClassifier,
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
