"""Kalman FC — pipeline completo: captura -> estabilização -> detecção -> Kalman -> vídeo.

Uso (a partir da raiz do repositório):

    python -m src.main videos/campo_lateral.MOV --save-video
    python -m src.main videos/campo_centro.MOV --start 30 --end 60 --show
    python -m src.main 0 --show                 # câmera ao vivo (índice 0)

Opções do filtro:  --sigma-a (px/s²)  --sigma-m (px)  --gate (prob. χ²)  --max-dark (quadros)

Saídas em outputs/<vídeo>/ (ou outputs/<--tag>/):
    track.csv       uma linha por quadro: estado, medição aceita, predição, estimativa, S, H
    track.json      parâmetros, desempenho e (se houver marcações) comparação com a verdade
    tracked.mp4     (com --save-video) vídeo com as sobreposições
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detector import BallDetector, DetectorParams
from src.evaluation import evaluate_track, load_annotations
from src.gating import chi2_threshold
from src.stabilize import Stabilizer, StabilizerParams
from src.tracker import BallTracker, TrackerParams
from src.video import frames, open_video, processing_size, video_info
from src.viz import Overlay


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="arquivo de vídeo ou índice da câmera")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--sigma-a", type=float, default=TrackerParams.sigma_a)
    ap.add_argument("--sigma-m", type=float, default=TrackerParams.sigma_m)
    ap.add_argument("--gate", type=float, default=TrackerParams.gate_prob)
    ap.add_argument("--max-dark", type=int, default=TrackerParams.max_dark)
    ap.add_argument("--no-stab", action="store_true", help="desliga a compensação da câmera")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--show", action="store_true", help="mostra a janela (q/ESC sai, espaço pausa)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default=None, help="nome da pasta de saída (padrão: nome do vídeo)")
    ap.add_argument("--annotations", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cap = open_video(args.video)
    info = video_info(cap, str(args.video))
    is_file = not str(args.video).isdigit()
    pw, ph = processing_size(info.width, info.height, args.width)
    start = int(args.start * info.fps)
    stop = int(args.end * info.fps) if args.end else (info.n_frames if is_file else None)
    name = args.tag or (Path(args.video).stem if is_file else f"camera{args.video}")
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tparams = TrackerParams(fps=info.fps, sigma_a=args.sigma_a, sigma_m=args.sigma_m,
                            gate_prob=args.gate, max_dark=args.max_dark)
    detector = BallDetector(DetectorParams())
    stab = None if args.no_stab else Stabilizer(StabilizerParams())
    tracker = BallTracker(tparams)
    overlay = Overlay(gate_threshold=chi2_threshold(args.gate))
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(out_dir / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (pw, ph))
    print(f"{info.path}: {info.width}x{info.height} -> {pw}x{ph}, fps={info.fps:.2f}, "
          f"sigma_a={args.sigma_a} sigma_m={args.sigma_m} gate={args.gate} max_dark={args.max_dark}"
          f"{' (sem estabilização)' if stab is None else ''}")

    rows = []
    t_total = 0.0
    paused = False
    for idx, frame in frames(cap, args.width, start, stop):
        t0 = time.perf_counter()
        H = stab.update(frame).H if stab is not None else np.eye(3)
        det = detector.detect(frame)
        cand_cur = np.array([[c.x, c.y] for c in det.candidates]).reshape(-1, 2)
        scores = np.array([c.score for c in det.candidates])
        cand_ref = Stabilizer.apply(H, cand_cur) if len(cand_cur) else cand_cur
        step = tracker.step(cand_ref, scores)
        step.frame = idx
        t_total += time.perf_counter() - t0

        est_cur = Stabilizer.apply(np.linalg.inv(H), step.x[:2])[0] if step.x is not None else (np.nan, np.nan)
        z_cur = cand_cur[step.z_index] if step.z_index is not None else (np.nan, np.nan)
        row = dict(frame=idx, t=idx / info.fps, state=step.state.value, n_cand=step.n_candidates,
                   z_x=z_cur[0], z_y=z_cur[1], est_x=est_cur[0], est_y=est_cur[1],
                   dark=step.dark, new_track=step.new_track, lost=step.lost)
        if step.z is not None:
            row.update(z_ref_x=step.z[0], z_ref_y=step.z[1])
        if step.x is not None:
            row.update(px=step.x[0], py=step.x[1], vx=step.x[2], vy=step.x[3])
        if step.z_pred is not None:
            row.update(pred_x=step.z_pred[0], pred_y=step.z_pred[1],
                       S_xx=step.S[0, 0], S_xy=step.S[0, 1], S_yy=step.S[1, 1])
        if step.d2 is not None and step.z_index is not None:
            row["d2"] = float(step.d2[step.z_index])
        row.update({f"h{i}{j}": H[i, j] for i in range(3) for j in range(3)})
        rows.append(row)

        if writer is not None or args.show:
            vis = overlay.draw(frame, step, H, cand_cur, info.fps)
            if writer is not None:
                writer.write(vis)
            if args.show:
                cv2.imshow("kalman-fc", vis)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    paused = not paused
        if is_file and (idx - start) % 600 == 0:
            print(f"  quadro {idx}/{stop}", flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    log = pd.DataFrame(rows)
    log.to_csv(out_dir / "track.csv", index=False)
    n = len(log)
    counts = log.state.value_counts(normalize=True).mul(100).round(1).to_dict()
    summary = {
        "video": info.path,
        "processing_size": [pw, ph],
        "params": {"sigma_a": args.sigma_a, "sigma_m": args.sigma_m, "gate_prob": args.gate,
                   "max_dark": args.max_dark, "stabilization": stab is not None},
        "frames": n,
        "pipeline_fps": round(n / t_total, 1) if t_total else None,
        "states_pct": counts,
        "tracks_started": int(log.new_track.sum()),
        "tracks_lost": int(log.lost.sum()),
    }
    ann_path = Path(args.annotations) if args.annotations else Path("data/annotations") / f"{Path(args.video).stem}.csv"
    if is_file and ann_path.exists():
        ann = load_annotations(ann_path, scale=pw / info.width)
        summary["vs_annotations"] = evaluate_track(log, ann)
    (out_dir / "track.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saídas em {out_dir}")


if __name__ == "__main__":
    main()
