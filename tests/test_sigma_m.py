"""experiments/sigma_m.py: estimativas de σm com a bola parada e em movimento."""
import numpy as np
import pandas as pd

from experiments.sigma_m import moving_sigma, static_sigma


def fake_track(sigma, n=600, seed=0, moving=True):
    rng = np.random.default_rng(seed)
    f = np.arange(n)
    t = f / 30
    # trajetória suave (parábolas por trechos não importam: a janela é curta)
    x = 100 + (60 * t + 20 * np.sin(t) if moving else 0 * t)
    y = 200 + (30 * t - 10 * np.cos(0.5 * t) if moving else 0 * t)
    nx, ny = rng.normal(0, sigma, n), rng.normal(0, sigma, n)
    return pd.DataFrame(dict(frame=f, ref_x=x + nx, ref_y=y + ny, cur_x=x + nx + 3, cur_y=y + ny))


def test_static_sigma_recovers_noise():
    r = static_sigma(fake_track(0.4, moving=False), 0, 599)
    assert abs(r["ref"]["std_x"] - 0.4) < 0.04 and abs(r["ref"]["std_y"] - 0.4) < 0.04


def test_moving_sigma_recovers_noise_on_smooth_trajectory():
    r = moving_sigma(fake_track(0.5, n=3000), win=7)
    assert abs(r["mad_x"] - 0.5) < 0.05 and abs(r["mad_y"] - 0.5) < 0.05


def test_moving_sigma_skips_windows_with_gaps():
    bt = fake_track(0.5, n=40)
    bt = bt[bt.frame % 4 != 0]               # nunca 7 quadros seguidos
    assert moving_sigma(bt, win=7)["n"] == 0
