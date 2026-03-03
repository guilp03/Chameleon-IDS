"""Testes para chameleon.data.preprocessing."""
import time

import numpy as np
import pandas as pd
import pytest

from chameleon.data.preprocessing import (
    get_time,
    normalize_data,
    particle_choices,
    split_train_val_test,
)


def _make_df(n_rows: int = 200, n_cols: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random((n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)])


def _make_labels(n_rows: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 2, n_rows))


# ---------------------------------------------------------------------------
# normalize_data
# ---------------------------------------------------------------------------

class TestNormalizeData:
    def test_returns_dataframe(self):
        df = _make_df()
        result = normalize_data(df)
        assert isinstance(result, pd.DataFrame)

    def test_mean_approx_zero(self):
        df = _make_df(n_rows=500)
        result = normalize_data(df)
        assert np.allclose(result.mean(), 0, atol=1e-10)

    def test_std_approx_one(self):
        df = _make_df(n_rows=500)
        result = normalize_data(df)
        assert np.allclose(result.std(ddof=0), 1, atol=1e-10)

    def test_columns_preserved(self):
        df = _make_df()
        result = normalize_data(df)
        assert list(result.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# split_train_val_test
# ---------------------------------------------------------------------------

class TestSplitTrainValTest:
    def setup_method(self):
        self.df = _make_df(n_rows=400)
        self.y = _make_labels(n_rows=400)
        self.column_names = list(self.df.columns)

    def test_six_return_values(self):
        result = split_train_val_test(self.df, self.column_names, self.y, test_size=0.2)
        assert len(result) == 6

    def test_train_size(self):
        x_train, y_train, *_ = split_train_val_test(
            self.df, self.column_names, self.y, test_size=0.2
        )
        assert len(x_train) == len(y_train)
        # 80 % dos dados para treino (±1 por arredondamento)
        assert abs(len(x_train) - int(0.8 * 400)) <= 1

    def test_val_test_equal_size(self):
        _, _, x_val, _, x_test, _ = split_train_val_test(
            self.df, self.column_names, self.y, test_size=0.2
        )
        # val e test são metade do test_size cada
        assert abs(len(x_val) - len(x_test)) <= 1

    def test_total_rows_preserved(self):
        x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(
            self.df, self.column_names, self.y, test_size=0.2
        )
        assert len(x_train) + len(x_val) + len(x_test) == 400


# ---------------------------------------------------------------------------
# get_time
# ---------------------------------------------------------------------------

class TestGetTime:
    def test_runs_without_error(self, capsys):
        get_time(0.0, 3661.0)  # 1h 1min 1s
        captured = capsys.readouterr()
        assert "horas" in captured.out
        assert "minutos" in captured.out
        assert "segundos" in captured.out

    def test_correct_hours_minutes_seconds(self, capsys):
        get_time(0.0, 3661.0)
        captured = capsys.readouterr()
        assert "1 horas" in captured.out
        assert "1 minutos" in captured.out
        assert "1 segundos" in captured.out

    def test_zero_time(self, capsys):
        get_time(100.0, 100.0)
        captured = capsys.readouterr()
        assert "0 horas" in captured.out


# ---------------------------------------------------------------------------
# particle_choices
# ---------------------------------------------------------------------------

class TestParticleChoices:
    def test_selects_correct_columns(self):
        pos = [0.2, 1, 0, 1, 0, 100]  # features 0 e 2 selecionadas
        column_names = ["a", "b", "c", "d"]
        result = particle_choices(pos, column_names, n_features=4)
        assert result == ["a", "c"]

    def test_none_selected(self):
        pos = [0.2, 0, 0, 0, 100]
        column_names = ["a", "b", "c"]
        result = particle_choices(pos, column_names, n_features=3)
        assert result == []

    def test_all_selected(self):
        pos = [0.2, 1, 1, 1, 100]
        column_names = ["a", "b", "c"]
        result = particle_choices(pos, column_names, n_features=3)
        assert result == ["a", "b", "c"]
