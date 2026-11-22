"""experiments/replay_tracker.py: a trilha repetida sobre candidatos guardados."""
import numpy as np

from experiments.replay_tracker import replay
from src.synthetic import SyntheticConfig, generate
from src.tracker import BallTracker, TrackerParams


def synthetic_cache(shift=(0.0, 0.0)):
    """Cache no formato de `collect`, com uma câmera que só translada (H conhecida)."""
    seq = generate(SyntheticConfig(n_frames=120, occlusions=[(50, 62)], seed=3,
                                   area=(4000, 4000), start=(2000, 2000)))
    H = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1.0]])
    Hinv = np.linalg.inv(H)
    # candidatos no quadro ATUAL = referência levada por H⁻¹
    cands = [c - np.array(shift) if len(c) else c for c in seq.candidates]
    return seq, dict(fps=30.0, frame=list(range(120)), H=[H] * 120, cands=cands,
                     scores=[np.ones(len(c)) for c in cands]), Hinv


def test_replay_f1_matches_direct_tracker_and_returns_to_current_frame():
    seq, data, _ = synthetic_cache(shift=(5.0, -3.0))
    log = replay(data, TrackerParams(sigma_a=800), field=None, metric=False)
    tr = BallTracker(TrackerParams(sigma_a=800))
    steps = [tr.step(c) for c in seq.candidates]                # direto, na referência
    assert list(log.state) == [s.state.value for s in steps]
    est_ref = np.array([s.x[:2] for s in steps])
    # o log guarda a estimativa no quadro atual: referência - deslocamento
    assert np.allclose(log[["est_x", "est_y"]].to_numpy(), est_ref - [5.0, -3.0])
    assert log.loc[55, "state"] == "NO_ESCURO" and np.isnan(log.loc[55, "z_x"])


def test_metric_defaults_have_physical_speed_limit():
    p = TrackerParams.metric()
    assert p.max_speed == 20.0 and p.max_dark == 20 and p.dark_damping == 1.0
    assert TrackerParams.metric(max_speed=1e9).max_speed == 1e9     # dá para desligar
    assert TrackerParams().max_speed == 1e9                          # F1 (px) sem limite


def test_replay_max_speed_limits_velocity():
    _, data, _ = synthetic_cache()
    log = replay(data, TrackerParams(sigma_a=4000, max_speed=50.0), field=None, metric=False)
    v = np.hypot(log.vx, log.vy).dropna()
    corrected = log.z_x.notna()
    assert (v[corrected] <= 50.0 + 1e-9).all()
