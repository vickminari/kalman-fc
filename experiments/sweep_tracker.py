"""Varredura de parâmetros da trilha (σa × probabilidade do gating) contra a verdade de campo.

Roda a estabilização e o detector uma vez, guarda os candidatos e depois executa o
rastreador para cada combinação, comparando com data/annotations/<vídeo>.csv.

Uso (a partir da raiz do repositório):

    python -m experiments.sweep_tracker videos/campo_lateral.MOV --end 30
    python -m experiments.sweep_tracker videos/campo_lateral.MOV --sigma-a 800 4000 --gate 0.99
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.detector import BallDetector
from src.evaluation import evaluate_track, load_annotations
from src.stabilize import Stabilizer
from src.tracker import BallTracker, TrackerParams
from src.video import frames, open_video, processing_size, video_info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--sigma-a", type=float, nargs="+", default=[800, 2000, 4000, 8000, 15000])
    ap.add_argument("--gate", type=float, nargs="+", default=[0.99, 0.999])
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--out", default=None, help="CSV com a tabela de resultados")
    args = ap.parse_args()

    cap = open_video(args.video)
    info = video_info(cap, args.video)
    pw, _ = processing_size(info.width, info.height, args.width)
    ann_path = Path(args.annotations) if args.annotations else Path("data/annotations") / f"{Path(args.video).stem}.csv"
    ann = load_annotations(ann_path, scale=pw / info.width)
    start = int(args.start * info.fps)
    stop = int(args.end * info.fps) if args.end else info.n_frames

    detector, stab = BallDetector(), Stabilizer()
    data = []
    for idx, f in frames(cap, args.width, start, stop):
        H = stab.update(f).H
        det = detector.detect(f)
        c = np.array([[q.x, q.y] for q in det.candidates]).reshape(-1, 2)
        s = np.array([q.score for q in det.candidates])
        data.append((idx, H, c, s))

    by_frame = {d[0]: d for d in data}
    vis = ann[(ann.visible == 1) & ann.frame.isin(by_frame)]
    hit = sum(1 for _, r in vis.iterrows()
              if len(by_frame[int(r.frame)][2])
              and np.hypot(*(by_frame[int(r.frame)][2] - [r.gx, r.gy]).T).min() <= args.radius)
    print(f"{len(data)} quadros; {len(vis)} marcados com a bola visível; "
          f"recall do detector = {100 * hit / max(len(vis), 1):.1f}%; "
          f"candidatos/quadro = {np.mean([len(d[2]) for d in data]):.2f}")

    results = []
    for sa in args.sigma_a:
        for g in args.gate:
            tr = BallTracker(TrackerParams(fps=info.fps, sigma_a=sa, gate_prob=g))
            rows = []
            for idx, H, c, s in data:
                cr = Stabilizer.apply(H, c) if len(c) else c
                st = tr.step(cr, s)
                e = Stabilizer.apply(np.linalg.inv(H), st.x[:2])[0] if st.x is not None else (np.nan, np.nan)
                z = c[st.z_index] if st.z_index is not None else (np.nan, np.nan)
                rows.append(dict(frame=idx, state=st.state.value, z_x=z[0], z_y=z[1], est_x=e[0], est_y=e[1]))
            log = pd.DataFrame(rows)
            r = evaluate_track(log, ann, radius=args.radius)
            res = dict(sigma_a=sa, gate=g, estimate_on_ball_pct=r["estimate_on_ball_pct"],
                       measurement_on_ball_pct=r["measurement_on_ball_pct"],
                       dark_pct=round(100 * float((log.state == "NO_ESCURO").mean()), 1),
                       accepted_while_invisible=r["accepted_while_invisible"],
                       error_estimate_mean=(r["error_estimate_px"] or {}).get("mean"))
            results.append(res)
            print("  ".join(f"{k}={v}" for k, v in res.items()))
    if args.out:
        pd.DataFrame(results).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
