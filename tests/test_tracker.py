"""Gating e lógica de trilha em sequências sintéticas."""
import numpy as np

from src.gating import chi2_threshold, ellipse, mahalanobis2, select
from src.synthetic import SyntheticConfig, generate
from src.tracker import BallTracker, TrackerParams, TrackState


# ------------------------------------------------------------------ gating
def test_chi2_thresholds():
    assert abs(chi2_threshold(0.95) - 5.991) < 1e-3
    assert abs(chi2_threshold(0.97) - (-2 * np.log(0.03))) < 1e-9


def test_gating_picks_nearest_in_mahalanobis_not_euclidean():
    S = np.array([[100.0, 0.0], [0.0, 1.0]])          # incerteza grande em x, pequena em y
    cands = np.array([[8.0, 0.0], [0.0, 3.0]])         # o 2º é mais perto em euclidiana
    i, d2 = select(cands, np.zeros(2), S, threshold=9.21)
    assert i == 0
    assert d2[0] < d2[1]


def test_gating_rejects_everything_outside():
    i, _ = select(np.array([[50.0, 50.0]]), np.zeros(2), np.eye(2) * 4, threshold=9.21)
    assert i is None


def test_ellipse_points_are_on_threshold():
    S = np.array([[30.0, 10.0], [10.0, 8.0]])
    pts = ellipse(np.array([5.0, -2.0]), S, 9.21)
    assert np.allclose(mahalanobis2(pts, np.array([5.0, -2.0]), S), 9.21)


# ------------------------------------------------------------------ trilha
def run(seq, params):
    tr = BallTracker(params)
    return [tr.step(c) for c in seq.candidates]


def test_tracker_coasts_through_occlusion_and_reacquires():
    cfg = SyntheticConfig(n_frames=200, occlusions=[(80, 100)], kick_every=(500, 600),
                          speed_range=(60, 60), seed=2, area=(4000, 4000), start=(2000, 2000))
    seq = generate(cfg)
    steps = run(seq, TrackerParams(sigma_a=100, max_dark=30))
    assert all(s.state == TrackState.DARK for s in steps[80:100])
    assert steps[100].state == TrackState.TRACKING and steps[100].z_index == seq.ball_index[100]
    # durante a oclusão, a predição segue a bola (velocidade constante)
    err = [np.linalg.norm(steps[t].x[:2] - seq.truth[t, :2]) for t in range(80, 100)]
    assert max(err) < 5.0


def test_tracker_loses_track_after_max_dark_and_restarts():
    cfg = SyntheticConfig(n_frames=120, occlusions=[(40, 80)], seed=4,
                          area=(4000, 4000), start=(2000, 2000))
    seq = generate(cfg)
    steps = run(seq, TrackerParams(max_dark=15))
    lost = [s.frame for s in steps if s.lost]
    assert lost == [40 + 15]
    assert steps[80].new_track
    assert steps[83].state == TrackState.TRACKING


def test_tracker_ignores_clutter_with_gating():
    cfg = SyntheticConfig(n_frames=300, clutter_per_frame=3.0, seed=7, kick_every=(60, 90))
    seq = generate(cfg)
    steps = run(seq, TrackerParams(sigma_a=800))
    tracked = [s for s in steps[10:] if s.z_index is not None]
    correct = sum(s.z_index == seq.ball_index[s.frame] for s in tracked)
    assert correct / len(tracked) > 0.95


def test_track_stuck_on_static_object_is_released():
    # a bola só aparece no quadro 150; antes, só uma "folha" parada, que inicializa a trilha
    cfg = SyntheticConfig(n_frames=220, occlusions=[(0, 150)], static_clutter=[(300.0, 300.0)],
                          seed=1)
    seq = generate(cfg)
    steps = run(seq, TrackerParams(static_max_s=2.0))
    # a trilha na folha é descartada quando surge uma alternativa (a bola)...
    assert any(s.lost and s.lost_reason == "parada" for s in steps[150:155])
    # ...e a próxima trilha confirmada segue a bola, não a folha
    s = next(s for s in steps[150:] if s.state == TrackState.TRACKING)
    assert np.linalg.norm(s.x[:2] - seq.truth[s.frame, :2]) < 10


def test_ball_at_rest_is_kept():
    # bola parada por 3 s com candidatos falsos aleatórios em volta: a trilha não pode cair
    cfg = SyntheticConfig(n_frames=90, speed_range=(0.0, 0.0), clutter_per_frame=2.0, seed=9)
    seq = generate(cfg)
    steps = run(seq, TrackerParams(static_max_s=4.0))
    # (a primeira trilha pode nascer num candidato falso e morrer como tentativa)
    first = next(s.frame for s in steps if s.state == TrackState.TRACKING)
    assert first < 20
    assert not any(s.lost for s in steps[first:])
    err = np.array([np.linalg.norm(s.x[:2] - seq.truth[s.frame, :2]) for s in steps[first:]])
    assert np.median(err) < 2.5 and err.max() < 10


def test_appearance_breaks_ties_inside_gate():
    tr = BallTracker(TrackerParams(confirm_hits=1))
    tr.step(np.array([[100.0, 100.0]]), np.array([0.9]))
    tr.step(np.array([[100.0, 100.0]]), np.array([0.9]))
    # dois candidatos dentro da região: um um pouco mais perto mas com cara de "pé"
    st = tr.step(np.array([[101.0, 100.0], [102.0, 100.5]]), np.array([0.05, 0.9]))
    assert st.z_index == 1
