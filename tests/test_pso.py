"""Testes para chameleon.pso.optimizer e chameleon.pso.particle."""
import pytest

from chameleon.pso.optimizer import clip_to_bounds, search_space, update_pb
from chameleon.pso.particle import Particle


def _make_particle(
    n_features: int = 5,
    ensemble_type: str = "gb",
    pb_val: float = 0.8,
    pos_val: float = 0.8,
) -> Particle:
    column_names = [f"f{i}" for i in range(n_features)]
    position = search_space(ensemble_type, n_features)
    p = Particle(0, position, ensemble_type=ensemble_type, column_names=column_names)
    p.pb_val = pb_val
    p.pos_val = pos_val
    return p


# ---------------------------------------------------------------------------
# search_space
# ---------------------------------------------------------------------------

class TestSearchSpace:
    def test_gb_position_length(self):
        n = 10
        pos = search_space("gb", n_features=n)
        # [test_size, feat_0..n-1, n_estimators, learning_rate]
        assert len(pos) == 1 + n + 2

    def test_rf_position_length(self):
        n = 10
        pos = search_space("rf", n_features=n)
        # [test_size, feat_0..n-1, n_estimators]
        assert len(pos) == 1 + n + 1

    def test_test_size_in_range(self):
        for _ in range(20):
            pos = search_space("gb", n_features=5)
            assert 0.1 <= pos[0] <= 0.4

    def test_feature_flags_binary(self):
        pos = search_space("gb", n_features=10)
        for flag in pos[1:11]:
            assert flag in (0, 1)

    def test_gb_n_estimators_in_range(self):
        for _ in range(20):
            pos = search_space("gb", n_features=5)
            assert 50 <= pos[-2] <= 1000

    def test_rf_n_estimators_in_range(self):
        for _ in range(20):
            pos = search_space("rf", n_features=5)
            assert 50 <= pos[-1] <= 600

    def test_gb_learning_rate_in_range(self):
        for _ in range(20):
            pos = search_space("gb", n_features=5)
            assert 0.1 <= pos[-1] <= 0.3


# ---------------------------------------------------------------------------
# clip_to_bounds
# ---------------------------------------------------------------------------

class TestClipToBounds:
    def test_test_size_clipped_below(self):
        p = _make_particle(n_features=5, ensemble_type="gb")
        p.position[0] = -0.5  # abaixo do mínimo
        clip_to_bounds(p, "gb", n_features=5)
        assert p.position[0] == pytest.approx(0.1)

    def test_test_size_clipped_above(self):
        p = _make_particle(n_features=5, ensemble_type="gb")
        p.position[0] = 0.99
        clip_to_bounds(p, "gb", n_features=5)
        assert p.position[0] == pytest.approx(0.4)

    def test_feature_binarization_above_half(self):
        p = _make_particle(n_features=5, ensemble_type="gb")
        p.position[1] = 0.7  # deve virar 1
        clip_to_bounds(p, "gb", n_features=5)
        assert p.position[1] == 1

    def test_feature_binarization_below_half(self):
        p = _make_particle(n_features=5, ensemble_type="gb")
        p.position[1] = 0.3  # deve virar 0
        clip_to_bounds(p, "gb", n_features=5)
        assert p.position[1] == 0

    def test_gb_n_estimators_clipped(self):
        p = _make_particle(n_features=5, ensemble_type="gb")
        p.position[-2] = 5000  # acima do máximo
        clip_to_bounds(p, "gb", n_features=5)
        assert p.position[-2] == 1000

    def test_rf_n_estimators_clipped_to_600(self):
        p = _make_particle(n_features=5, ensemble_type="rf")
        p.position[-1] = 9999
        clip_to_bounds(p, "rf", n_features=5)
        assert p.position[-1] == 600

    def test_gb_learning_rate_clipped(self):
        p = _make_particle(n_features=5, ensemble_type="gb")
        p.position[-1] = 0.99  # acima do máximo
        clip_to_bounds(p, "gb", n_features=5)
        assert p.position[-1] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# update_pb
# ---------------------------------------------------------------------------

class TestUpdatePb:
    def test_pb_updated_when_better(self):
        p = _make_particle(n_features=5, ensemble_type="gb", pb_val=0.7, pos_val=0.9)
        old_position = list(p.position)
        update_pb(p)
        assert p.pb_val == pytest.approx(0.9)
        assert p.personal_best == p.position

    def test_pb_not_updated_when_equal(self):
        p = _make_particle(n_features=5, ensemble_type="gb", pb_val=0.8, pos_val=0.8)
        p.personal_best = [99.0]  # sentinela para verificar que não mudou
        update_pb(p)
        assert p.pb_val == pytest.approx(0.8)
        assert p.personal_best == [99.0]

    def test_pb_not_updated_when_worse(self):
        p = _make_particle(n_features=5, ensemble_type="gb", pb_val=0.9, pos_val=0.5)
        p.personal_best = [99.0]
        update_pb(p)
        assert p.pb_val == pytest.approx(0.9)
        assert p.personal_best == [99.0]

    def test_returns_particle(self):
        p = _make_particle()
        result = update_pb(p)
        assert result is p
