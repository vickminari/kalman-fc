"""Comparação do rastreamento com a verdade de campo marcada à mão."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_annotations(path: Path, scale: float) -> pd.DataFrame:
    """Lê data/annotations/<vídeo>.csv e converte x, y para px de processamento."""
    ann = pd.read_csv(path)
    ann["gx"] = ann.x * scale
    ann["gy"] = ann.y * scale
    return ann


def evaluate_track(log: pd.DataFrame, ann: pd.DataFrame, radius: float = 10.0) -> dict:
    """Erros da medição aceita (detector) e da estimativa (Kalman) contra as marcações.

    Tudo em px de processamento, no quadro atual. `radius` define "acertou a bola".
    """
    m = ann.merge(log, on="frame", how="inner")
    if m.empty:
        return {"annotated_frames_in_range": 0}
    vis = m[m.visible == 1]
    inv = m[m.visible == 0]

    def dist(df, xc, yc):
        return np.hypot(df[xc] - df.gx, df[yc] - df.gy)

    meas = vis.dropna(subset=["z_x"])
    est = vis.dropna(subset=["est_x"])
    d_meas = dist(meas, "z_x", "z_y")
    d_est = dist(est, "est_x", "est_y")
    dark = vis[vis.state == "NO_ESCURO"].dropna(subset=["est_x"])
    d_dark = dist(dark, "est_x", "est_y")

    def stats(d):
        return None if len(d) == 0 else {
            "n": int(len(d)), "mean": round(float(d.mean()), 2),
            "median": round(float(d.median()), 2), "p90": round(float(d.quantile(.9)), 2)}

    nv = len(vis)
    return {
        "annotated_frames_in_range": int(len(m)),
        "ball_visible": int(nv),
        "ball_not_visible": int(len(inv)),
        "estimate_on_ball_pct": round(100 * float((d_est <= radius).sum()) / nv, 1) if nv else None,
        "measurement_on_ball_pct": round(100 * float((d_meas <= radius).sum()) / nv, 1) if nv else None,
        "estimate_off_ball_pct": round(100 * float((d_est > radius).sum()) / nv, 1) if nv else None,
        "accepted_while_invisible": int(inv.z_x.notna().sum()),
        "error_measurement_px": stats(d_meas[d_meas <= radius]),
        "error_estimate_px": stats(d_est[d_est <= radius]),
        "error_dark_prediction_px": stats(d_dark),
        "radius_px": radius,
    }
