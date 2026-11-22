"""experiments/e2_sigma_a.py: atraso medido contra marcações e escolha do trecho."""
import numpy as np
import pandas as pd

from experiments.e2_sigma_a import lag_frames, pick_segment, raw_vs_filtered


def moving_ball(n=300, lag=0):
    """Bola em zigue-zague (5 px/quadro em x, y senoidal); a estimativa está atrasada `lag` quadros."""
    f = np.arange(n)
    x = 50 + 5 * f % 500
    y = 180 + 40 * np.sin(f / 15)
    ex, ey = np.roll(x, lag), np.roll(y, lag)
    log = pd.DataFrame(dict(frame=f, est_x=ex.astype(float), est_y=ey, z_x=x.astype(float), z_y=y,
                            state="RASTREANDO"))
    marks = f[::5]
    annr = pd.DataFrame(dict(frame=marks, rx=x[marks].astype(float), ry=y[marks]))
    v = np.gradient(annr[["rx", "ry"]].to_numpy(), axis=0) / 5
    annr["vx"], annr["vy"] = v[:, 0], v[:, 1]
    annr["speed"] = np.hypot(v[:, 0], v[:, 1])
    return log, annr


def test_lag_frames_recovers_known_delay():
    for lag in (0, 2):
        log, annr = moving_ball(lag=lag)
        r = lag_frames(log, annr, on_ball=60)
        assert abs(r["lag_frames"] - lag) < 0.3, (lag, r)


def test_raw_vs_filtered_uses_same_frames():
    log, annr = moving_ball()
    ann = annr.rename(columns={"rx": "gx", "ry": "gy"}).assign(visible=1)
    r = raw_vs_filtered(log, ann)
    assert r["n"] == len(ann) and r["meas_median_px"] == 0 and r["est_median_px"] == 0


def test_pick_segment_finds_the_kick():
    log, annr = moving_ball(n=300)
    # um "chute": a velocidade marcada muda bruscamente no quadro 200
    annr.loc[annr.frame == 200, ["vx", "vy"]] = [20.0, 0.0]
    f0, f1 = pick_segment(log, annr, length=40)
    assert f0 <= 200 < f1
