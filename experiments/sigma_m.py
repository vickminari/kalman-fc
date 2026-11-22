"""Mede o ruído da medição σm (px a 640) nos vídeos de teste (T20, Tarefa 5 da limpeza).

Duas estimativas, ambas no QUADRO DE REFERÊNCIA (é onde o filtro F1 recebe a medição, então
o erro da estabilização faz parte do ruído que o Σm precisa representar):

1. **Bola parada:** nos quadros em que a bola não se mexe (lateral: ~0–100), a posição do
   candidato-bola (o mais próximo da marcação, a ≤ `radius` px) deveria ser constante. O
   desvio-padrão por eixo é σm. Também em px do quadro atual, para ver quanto o balanço da
   câmera acrescentaria se não houvesse estabilização.
2. **Bola em movimento:** em trechos de quadros seguidos com o candidato-bola, ajusta uma
   parábola por eixo numa janela deslizante (`win` quadros; aceleração constante na janela)
   e usa o resíduo. O desvio do resíduo, corrigido pelos graus de liberdade da janela, é
   outra estimativa de σm, que inclui o borrão de movimento. Chutes violam a parábola, por
   isso usamos uma estimativa robusta (MAD).

Usa o cache de candidatos de `experiments/replay_tracker.py` (gera se não existir).

Uso (a partir da raiz do repositório):

    python -m experiments.sigma_m videos/campo_lateral.MOV --static 0 100
    python -m experiments.sigma_m videos/campo_centro.MOV
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.replay_tracker import collect
from src.evaluation import load_annotations
from src.stabilize import Stabilizer


def ball_track(data: dict, ann: pd.DataFrame, radius: float = 10.0) -> pd.DataFrame:
    """Candidato-bola por quadro: o mais próximo da marcação, interpolada entre quadros marcados.

    As marcações são a cada ~5 quadros; entre elas, interpolamos linearmente a posição só
    para ESCOLHER o candidato (com um raio folgado); a medida usada é a do detector.
    Só entram trechos entre dois quadros marcados com a bola visível.
    """
    vis = ann[ann.visible == 1].sort_values("frame")
    f = np.array(data["frame"])
    gx = np.interp(f, vis.frame, vis.gx, left=np.nan, right=np.nan)
    gy = np.interp(f, vis.frame, vis.gy, left=np.nan, right=np.nan)
    # quadros entre duas marcações visíveis consecutivas (sem uma invisível no meio)
    marks = ann.sort_values("frame")
    ok = np.zeros(len(f), bool)
    for (fa, va), (fb, vb) in zip(marks[["frame", "visible"]].values[:-1], marks[["frame", "visible"]].values[1:]):
        if va == 1 and vb == 1 and fb - fa <= 10:
            ok[(f >= fa) & (f <= fb)] = True
    rows = []
    for i, (idx, H, c) in enumerate(zip(f, data["H"], data["cands"])):
        if not ok[i] or len(c) == 0 or np.isnan(gx[i]):
            continue
        d = np.hypot(c[:, 0] - gx[i], c[:, 1] - gy[i])
        k = int(np.argmin(d))
        if d[k] > radius:
            continue
        r = Stabilizer.apply(H, c[k:k + 1])[0]
        rows.append(dict(frame=int(idx), cur_x=c[k, 0], cur_y=c[k, 1], ref_x=r[0], ref_y=r[1]))
    return pd.DataFrame(rows)


def static_sigma(bt: pd.DataFrame, f0: int, f1: int) -> dict:
    """Desvio-padrão por eixo da posição do candidato-bola com a bola parada."""
    s = bt[(bt.frame >= f0) & (bt.frame <= f1)]
    out = {"frames": [f0, f1], "n": int(len(s))}
    for name in ("ref", "cur"):
        xy = s[[f"{name}_x", f"{name}_y"]].to_numpy()
        sd = xy.std(axis=0, ddof=1)
        mad = 1.4826 * np.median(np.abs(xy - np.median(xy, axis=0)), axis=0)
        out[name] = {"std_x": round(float(sd[0]), 3), "std_y": round(float(sd[1]), 3),
                     "mad_x": round(float(mad[0]), 3), "mad_y": round(float(mad[1]), 3)}
    return out


def moving_sigma(bt: pd.DataFrame, win: int = 7, exclude: tuple[int, int] | None = None) -> dict:
    """σm pelo resíduo de parábolas locais (janela de `win` quadros seguidos com medição).

    Para ruído branco de desvio σ, o resíduo no centro de uma janela de N pontos ajustada por
    um polinômio de grau 2 tem variância σ²(1 − h), com h a alavanca do ponto central; o
    resíduo é dividido por √(1 − h) para estimar σ.
    """
    if exclude is not None:
        bt = bt[(bt.frame < exclude[0]) | (bt.frame > exclude[1])]
    f = bt.frame.to_numpy()
    xy = bt[["ref_x", "ref_y"]].to_numpy()
    half = win // 2
    t = np.arange(-half, half + 1, dtype=float)
    A = np.column_stack([np.ones_like(t), t, t ** 2])
    Hat = A @ np.linalg.pinv(A)
    h = Hat[half, half]
    res = []
    for i in range(half, len(f) - half):
        if f[i + half] - f[i - half] != 2 * half:      # janela com quadros seguidos
            continue
        w = xy[i - half:i + half + 1]
        fit = Hat[half] @ w
        res.append((xy[i] - fit) / np.sqrt(1 - h))
    res = np.array(res)
    if len(res) == 0:
        return {"n": 0}
    mad = 1.4826 * np.median(np.abs(res - np.median(res, axis=0)), axis=0)
    return {"window": win, "n": int(len(res)),
            "mad_x": round(float(mad[0]), 3), "mad_y": round(float(mad[1]), 3),
            "std_x": round(float(res[:, 0].std()), 3), "std_y": round(float(res[:, 1].std()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--static", type=int, nargs=2, default=None, help="quadros [ini fim] com a bola parada")
    ap.add_argument("--win", type=int, default=7)
    ap.add_argument("--radius", type=float, default=10.0)
    args = ap.parse_args()

    stem = Path(args.video).stem
    scorer = args.scorer or f"data/models/ball_scorer_{stem}.json"
    exp_dir = Path("outputs/experiments") / stem
    data = collect(args.video, scorer, cache=exp_dir / "replay_cache.pkl")
    ann = load_annotations(Path("data/annotations") / f"{stem}.csv", scale=data["size"][0] / data["orig_width"])
    bt = ball_track(data, ann, args.radius)
    out = {"video": args.video, "ball_frames": int(len(bt))}
    if args.static:
        out["static"] = static_sigma(bt, *args.static)
    out["moving"] = moving_sigma(bt, args.win, exclude=tuple(args.static) if args.static else None)
    (exp_dir / "sigma_m.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
