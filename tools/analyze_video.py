"""Análise de um vídeo de teste: detector quadro a quadro + movimento da câmera.

Uso (a partir da raiz do repositório):

    python -m tools.analyze_video videos/campo_lateral.mp4
    python -m tools.analyze_video videos/campo_centro.mp4 --save-video --start 30 --end 60

Saídas em outputs/<nome do vídeo>/:
    frames.csv       uma linha por quadro (melhor candidato, nº de candidatos, movimento da câmera)
    candidates.csv   todos os candidatos de todos os quadros
    summary.json     estatísticas gerais
    plots.png        candidatos, movimento da câmera e tamanho da bola × posição vertical
    debug_grid.jpg   quadros amostrados com as máscaras intermediárias
    overlay.mp4      (com --save-video) vídeo com a borda do campo e os candidatos
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detector import BallDetector, DetectorParams, draw_candidates
from src.motion import CameraMotion
from src.video import frames, open_video, processing_size, video_info


def build_debug_grid(samples: list[tuple[int, np.ndarray]], detector: BallDetector,
                     fps: float, tile_w: int = 320) -> np.ndarray:
    rows = []
    for idx, frame in samples:
        det = detector.detect(frame, keep_masks=True)
        over = draw_candidates(frame, det)
        tiles = [over]
        for name in ("grass", "white", "candidates"):
            m = cv2.cvtColor(det.masks[name], cv2.COLOR_GRAY2BGR)
            if name == "white":  # mostra a região do campo escurecida por baixo
                f = cv2.cvtColor(det.masks["field"], cv2.COLOR_GRAY2BGR)
                m = cv2.addWeighted(m, 1.0, f, 0.25, 0)
            tiles.append(m)
        h = int(frame.shape[0] * tile_w / frame.shape[1])
        tiles = [cv2.resize(t, (tile_w, h), interpolation=cv2.INTER_AREA) for t in tiles]
        labels = [f"quadro {idx} ({idx / fps:.1f}s) - {len(det.candidates)} cand.",
                  "grama", "branco + campo", "nao-grama no campo (pos-morfologia)"]
        for t, lab in zip(tiles, labels):
            cv2.putText(t, lab, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack(tiles))
    return np.vstack(rows)


def evaluate(cands: pd.DataFrame, df: pd.DataFrame, ann_path: Path, scale: float,
             radius: float) -> dict:
    """Compara os candidatos do detector com as marcações manuais.

    - recall: em quantos quadros com a bola visível algum candidato está a até `radius` px;
    - best_is_ball: em quantos o candidato de maior score é a bola;
    - false_per_frame: candidatos que não são a bola, por quadro marcado;
    - erro do centroide (px de processamento) quando a bola é achada.
    """
    ann = pd.read_csv(ann_path)
    ann = ann[ann.frame.isin(df.frame)]
    vis = ann[ann.visible == 1]
    by_frame = {f: g for f, g in cands.groupby("frame")}
    hits, best_hits, errs, false = 0, 0, [], 0
    for _, a in ann.iterrows():
        g = by_frame.get(int(a.frame))
        n = 0 if g is None else len(g)
        if a.visible != 1:
            false += n
            continue
        if g is None:
            continue
        d = np.hypot(g.x - a.x * scale, g.y - a.y * scale)
        ok = d <= radius
        false += int((~ok).sum())
        if ok.any():
            hits += 1
            errs.append(float(d[ok].min()))
            if d.iloc[int(np.argmax(g.score.values))] <= radius:
                best_hits += 1
    nv = len(vis)
    return {
        "annotated_frames": int(len(ann)),
        "ball_visible_frames": int(nv),
        "recall_pct": round(100 * hits / nv, 1) if nv else None,
        "best_is_ball_pct": round(100 * best_hits / nv, 1) if nv else None,
        "false_candidates_per_frame": round(false / max(len(ann), 1), 2),
        "centroid_error_px_median": round(float(np.median(errs)), 2) if errs else None,
    }


def make_plots(df: pd.DataFrame, cands: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    ax[0].plot(df.t, df.n_cand, lw=0.6)
    ax[0].set_ylabel("nº de candidatos")
    ax[0].set_title(title)
    ax[1].plot(df.t, df.cam_dx.cumsum(), label="x acumulado", lw=0.8)
    ax[1].plot(df.t, df.cam_dy.cumsum(), label="y acumulado", lw=0.8)
    ax[1].set_ylabel("movimento da câmera (px)")
    ax[1].legend(loc="upper left")
    ax[2].scatter(cands.y, cands.diameter, s=2, alpha=0.3, label="todos os candidatos")
    best = df.dropna(subset=["best_y"])
    ax[2].scatter(best.best_y, best.best_d, s=3, alpha=0.5, c="tab:red", label="melhor por quadro")
    ax[2].set_xlabel("posição vertical na imagem (px)  — topo = fundo do campo")
    ax[2].set_ylabel("diâmetro equivalente (px)")
    ax[2].legend(loc="upper left")
    ax[0].set_xlabel("tempo (s)")
    ax[1].set_xlabel("tempo (s)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--width", type=int, default=640, help="largura de processamento (padrão 640)")
    ap.add_argument("--start", type=float, default=0.0, help="início em segundos")
    ap.add_argument("--end", type=float, default=None, help="fim em segundos")
    ap.add_argument("--save-video", action="store_true", help="gravar overlay.mp4")
    ap.add_argument("--samples", type=int, default=6, help="quadros no debug_grid.jpg")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--annotations", default=None,
                    help="CSV de marcações (padrão data/annotations/<vídeo>.csv, se existir)")
    ap.add_argument("--match-radius", type=float, default=8.0,
                    help="distância máx. (px de processamento) para um candidato contar como a bola")
    args = ap.parse_args()

    cap = open_video(args.video)
    info = video_info(cap, args.video)
    pw, ph = processing_size(info.width, info.height, args.width)
    start = int(args.start * info.fps)
    stop = int(args.end * info.fps) if args.end else info.n_frames
    out_dir = Path(args.out) / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{info.path}: {info.width}x{info.height} @ {info.fps:.2f} fps, "
          f"{info.n_frames} quadros ({info.duration:.1f}s) -> processando em {pw}x{ph}")

    detector = BallDetector(DetectorParams())
    motion = CameraMotion()
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (pw, ph))

    sample_idx = set(np.linspace(start, stop - 1, args.samples).astype(int).tolist())
    samples: list[tuple[int, np.ndarray]] = []
    rows, cand_rows = [], []
    t_det = 0.0
    for idx, frame in frames(cap, args.width, start, stop):
        dx, dy, resp = motion.update(frame)
        t0 = time.perf_counter()
        det = detector.detect(frame)
        dt = time.perf_counter() - t0
        t_det += dt
        b = det.best
        rows.append(dict(frame=idx, t=idx / info.fps, n_cand=len(det.candidates),
                         best_x=b.x if b else np.nan, best_y=b.y if b else np.nan,
                         best_d=b.diameter if b else np.nan, best_score=b.score if b else np.nan,
                         cam_dx=dx, cam_dy=dy, cam_resp=resp, det_ms=dt * 1000))
        for c in det.candidates:
            cand_rows.append(dict(frame=idx, x=c.x, y=c.y, area=c.area, diameter=c.diameter,
                                  w=c.w, h=c.h, circularity=c.circularity, whiteness=c.whiteness,
                                  ring_grass=c.ring_grass, score=c.score))
        if idx in sample_idx:
            samples.append((idx, frame.copy()))
        if writer is not None:
            over = draw_candidates(frame, det)
            cv2.putText(over, f"{idx} {idx / info.fps:6.2f}s  cand={len(det.candidates)}  "
                              f"cam=({dx:+.1f},{dy:+.1f})", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
            writer.write(over)
        if (idx - start) % 300 == 0:
            print(f"  quadro {idx}/{stop}", flush=True)
    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    cands = pd.DataFrame(cand_rows, columns=["frame", "x", "y", "area", "diameter", "w", "h",
                                             "circularity", "whiteness", "ring_grass", "score"])
    df.to_csv(out_dir / "frames.csv", index=False)
    cands.to_csv(out_dir / "candidates.csv", index=False)

    step = np.hypot(df.cam_dx, df.cam_dy)
    n = len(df)
    summary = {
        "video": info.path,
        "original_size": [info.width, info.height],
        "processing_size": [pw, ph],
        "fps": info.fps,
        "frames_analyzed": n,
        "frames_with_candidate_pct": round(100 * float((df.n_cand > 0).mean()), 1),
        "frames_with_one_candidate_pct": round(100 * float((df.n_cand == 1).mean()), 1),
        "frames_with_multiple_candidates_pct": round(100 * float((df.n_cand > 1).mean()), 1),
        "mean_candidates": round(float(df.n_cand.mean()), 2),
        "best_diameter_px": {q: round(float(df.best_d.quantile(v)), 1)
                             for q, v in (("p5", .05), ("median", .5), ("p95", .95))}
        if df.best_d.notna().any() else None,
        "detector_ms_mean": round(t_det * 1000 / max(n, 1), 2),
        "detector_fps": round(n / t_det, 1) if t_det else None,
        "camera_motion": {
            "median_px_per_frame": round(float(step.median()), 2),
            "p95_px_per_frame": round(float(step.quantile(.95)), 2),
            "frames_moving_pct (>0.5px)": round(100 * float((step > 0.5).mean()), 1),
            "cumulative_x_range_px": [round(float(df.cam_dx.cumsum().min())),
                                      round(float(df.cam_dx.cumsum().max()))],
            "cumulative_y_range_px": [round(float(df.cam_dy.cumsum().min())),
                                      round(float(df.cam_dy.cumsum().max()))],
        },
    }
    ann_path = Path(args.annotations) if args.annotations else \
        Path("data/annotations") / f"{Path(args.video).stem}.csv"
    if ann_path.exists():
        summary["vs_annotations"] = evaluate(cands, df, ann_path, scale=pw / info.width,
                                             radius=args.match_radius)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    make_plots(df, cands, out_dir / "plots.png", Path(args.video).name)
    if samples:
        cv2.imwrite(str(out_dir / "debug_grid.jpg"), build_debug_grid(samples, BallDetector(DetectorParams()), info.fps))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saídas em {out_dir}")


if __name__ == "__main__":
    main()
