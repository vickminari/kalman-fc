"""Ganho de Kalman em regime e o script de diagnóstico da trilha (experiments/track_diagnostics.py)."""
import numpy as np
import pandas as pd

from experiments.track_diagnostics import dark_episodes, diagnose, jumps, to_ref
from src.kalman import KalmanFilter, constant_velocity, init_state, steady_state_gain


# ------------------------------------------------------------------ ganho em regime
def test_steady_state_gain_matches_table_of_report_010():
    """K da posição para σm = 1,5 px a 30 fps: a tabela do report 010."""
    expected = {100: 0.32, 800: 0.66, 4000: 0.90, 15000: 0.98}
    for sa, k in expected.items():
        m = constant_velocity(1 / 30, sa, 1.5)
        K, _ = steady_state_gain(m.D, m.M, m.Sd, m.Sm)
        assert abs(K[0, 0] - k) < 0.005, (sa, K[0, 0])
        assert abs(K[1, 1] - k) < 0.005


def test_steady_state_gain_equals_gain_of_filter_after_many_steps():
    m = constant_velocity(1 / 30, 800, 1.5)
    K_ss, _ = steady_state_gain(m.D, m.M, m.Sd, m.Sm)
    x0, P0 = init_state(np.zeros(2), 1.5, 300)
    kf = KalmanFilter(m.D, m.M, m.Sd, m.Sm, x0, P0)
    for _ in range(300):
        kf.predict()
        kf.correct(np.zeros(2))
    kf.predict()
    K = kf.P_pred @ m.M.T @ np.linalg.inv(m.M @ kf.P_pred @ m.M.T + m.Sm)
    assert np.allclose(K, K_ss, atol=1e-6)


# ------------------------------------------------------------------ diagnóstico
def make_log():
    """Trilha de brinquedo: medição em 0..4, escuro em 5..7 com a predição "fugindo",
    retomada longe em 8, quadro 9 normal, trilha nova longe em 10."""
    est = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),          # com medição
           (30, 0), (60, 0), (90, 0),                         # no escuro, 30 px/quadro
           (10, 0), (11, 0),                                  # retomada (salto de 80) e normal
           (200, 200)]                                        # trilha nova
    state = ["TENTATIVA"] + ["RASTREANDO"] * 4 + ["NO_ESCURO"] * 3 + ["RASTREANDO"] * 2 + ["TENTATIVA"]
    has_z = [True] * 5 + [False] * 3 + [True] * 3
    dark = [0] * 5 + [1, 2, 3] + [0] * 3
    new = [True] + [False] * 9 + [True]
    rows = []
    for f, ((x, y), s, hz, d, n) in enumerate(zip(est, state, has_z, dark, new)):
        rows.append(dict(frame=f, t=f / 30, state=s, est_x=float(x), est_y=float(y),
                         z_x=float(x) if hz else np.nan, z_y=float(y) if hz else np.nan,
                         dark=d, new_track=n, lost=False, speed_kmh=10.0))
    return pd.DataFrame(rows)


def test_to_ref_applies_homography_per_row():
    log = make_log()
    for k, v in zip([f"h{i}{j}" for i in range(3) for j in range(3)], np.eye(3).ravel()):
        log[k] = v
    log["h02"] = 5.0                                          # translação de 5 px em x
    p = to_ref(log, "est_x", "est_y")
    assert np.allclose(p[:, 0], log.est_x + 5)
    assert np.isnan(to_ref(log, "z_x", "z_y")[5]).all()       # sem medição continua NaN


def test_jumps_are_classified_by_cause():
    log = make_log()
    j = jumps(log, log[["est_x", "est_y"]].to_numpy(float), jump_px=20)
    assert j["total"] == 5
    assert j["no_escuro"] == 3 and j["retomada"] == 1 and j["trilha_nova"] == 1 and j["outro"] == 0
    assert j["dark_step_px"]["median"] == 30.0


def test_dark_episode_recovered():
    de = dark_episodes(make_log())
    assert de["episodes"] == 1 and de["recovered"] == 1 and de["lost"] == 0
    assert de["recovered_len"]["median"] == 3


def test_diagnose_association_and_reference_metrics():
    log = make_log()
    # marcações: bola em (x, 0) nos quadros 0..4 e 8..9; invisível em 10
    ann = pd.DataFrame(dict(frame=[0, 2, 4, 6, 8, 10], visible=[1, 1, 1, 1, 1, 0],
                            gx=[0.0, 2.0, 4.0, 40.0, 10.0, 0.0], gy=[0.0] * 6))
    r = diagnose(log, ann, fps=30, sigma_a=4000, Sm=np.eye(2) * 2.25, split=0)
    a = r["association"]
    assert a["accepted_visible"] == 4 and a["accepted_on_ball"] == 4
    assert a["accepted_invisible"] == 1                       # quadro 10, trilha nova sem bola
    assert a["visible_without_measurement"] == 1               # quadro 6, no escuro
    assert r["reference"]["on_ball_all_pct"] == 80.0           # 4 de 5 (o 6 erra 20 px)
    assert r["reference"]["dark_pred_err_all_median_px"] == 20.0
    assert abs(r["kalman_gain"]["K_pos_min"] - 0.90) < 0.01
    assert r["est_to_meas_px"]["median"] == 0.0
