"""Testes para chameleon.models.autoencoder."""
import numpy as np
import pytest
import torch

from chameleon.models.autoencoder import Autoencoder, EarlyStopping, get_overall_metrics


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_triggers_after_patience_exceeded(self, tmp_path):
        es = EarlyStopping(patience=3, delta=0.0, verbose=False, path=str(tmp_path / "ckpt.pt"))
        model = Autoencoder(4)
        model.compile(learning_rate=1e-3)

        es(1.0, model)  # salva checkpoint (melhora)
        assert not es.early_stop

        for _ in range(3):
            es(1.1, model)  # sem melhora

        assert es.early_stop

    def test_counter_resets_on_improvement(self, tmp_path):
        es = EarlyStopping(patience=3, delta=0.0, verbose=False, path=str(tmp_path / "ckpt.pt"))
        model = Autoencoder(4)
        model.compile(learning_rate=1e-3)

        es(1.0, model)  # melhora
        es(1.1, model)  # piora — counter=1
        es(0.5, model)  # melhora — counter deve resetar para 0
        assert es.counter == 0

    def test_not_triggered_before_patience(self, tmp_path):
        es = EarlyStopping(patience=5, delta=0.0, verbose=False, path=str(tmp_path / "ckpt.pt"))
        model = Autoencoder(4)
        model.compile(learning_rate=1e-3)

        es(1.0, model)
        for _ in range(4):  # 4 < patience=5
            es(2.0, model)
        assert not es.early_stop

    def test_saves_checkpoint(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        es = EarlyStopping(patience=3, delta=0.0, verbose=False, path=str(ckpt))
        model = Autoencoder(4)
        model.compile(learning_rate=1e-3)
        es(0.5, model)
        assert ckpt.exists()


# ---------------------------------------------------------------------------
# get_overall_metrics
# ---------------------------------------------------------------------------

class TestGetOverallMetrics:
    def _perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        return get_overall_metrics(y_true, y_pred)

    def test_returns_dict(self):
        result = self._perfect()
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = self._perfect()
        expected_keys = {"accuracy", "tpr", "fpr", "precision", "f1-score", "recall"}
        assert set(result.keys()) == expected_keys

    def test_perfect_predictions(self):
        result = self._perfect()
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["tpr"] == pytest.approx(1.0)
        assert result["fpr"] == pytest.approx(0.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["f1-score"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = get_overall_metrics(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(0.0)
        assert result["tpr"] == pytest.approx(0.0)
        assert result["fpr"] == pytest.approx(1.0)

    def test_values_between_zero_and_one(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = rng.integers(0, 2, 100)
        # Garante que há pelo menos um TP, TN, FP, FN para evitar divisão por zero
        y_true[:25] = 0
        y_true[25:50] = 1
        y_pred[:25] = 0
        y_pred[25:50] = 1
        result = get_overall_metrics(y_true, y_pred)
        for value in result.values():
            assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Autoencoder — forward pass
# ---------------------------------------------------------------------------

class TestAutoencoderForward:
    def test_output_shape(self):
        model = Autoencoder(in_features=10)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 10)

    def test_output_in_zero_one_range(self):
        model = Autoencoder(in_features=10)
        x = torch.randn(8, 10)
        out = model(x)
        assert (out >= 0).all() and (out <= 1).all()
