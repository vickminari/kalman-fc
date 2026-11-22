"""Estabilização de um vídeo gravado com a câmera na mão (verificação visual e estatísticas).

Uso (a partir da raiz do repositório):

    python -m tools.stabilize_video videos/campo_lateral.MOV --save-video
    python -m tools.stabilize_video videos/campo_centro.MOV --start 30 --end 60 --save-video

Saídas em outputs/<vídeo>/:
    stabilization.csv     H (quadro -> referência) por quadro, inliers, correções
    stabilization.json    estatísticas
    stabilized.mp4        (com --save-video) cada quadro projetado no plano da
                          referência sobre um mosaico do campo. Se a estabilização
                          estiver boa, a parede, o gol e a grama ficam parados e só
                          os jogadores e a bola se movem.
    mosaic.jpg            mosaico de todos os quadros no plano da referência
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.stabilize import Stabilizer, StabilizerParams
from src.video import frames, open_video, processing_size, video_info


def canvas_from(Hs: list[np.ndarray], w: int, h: int, max_side: int = 4000):
    """Translação + tamanho do canvas que cobre todos os quadros projetados."""
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    allc = np.vstack([cv2.perspectiveTransform(corners, H).reshape(-1, 2) for H in Hs])
    lo = np.percentile(allc, 1, axis=0) - 10
    hi = np.percentile(allc, 99, axis=0) + 10
    lo = np.maximum(lo, [-2 * w, -2 * h])
    hi = np.minimum(hi, [3 * w, 3 * h])
    size = np.ceil(hi - lo).astype(int)
    s = min(1.0, max_side / size.max())
    T = np.array([[s, 0, -lo[0] * s], [0, s, -lo[1] * s], [0, 0, 1]])
    return T, (int(size[0] * s), int(size[1] * s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--width", type=int, default=640, help="largura de processamento (padrão 640)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cap = open_video(args.video)
    info = video_info(cap, args.video)
    pw, ph = processing_size(info.width, info.height, args.width)
    start = int(args.start * info.fps)
    stop = int(args.end * info.fps) if args.end else info.n_frames
    out_dir = Path(args.out) / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{info.path}: {info.width}x{info.height}, {info.n_frames} quadros -> {pw}x{ph}, "
          f"referência = quadro {start}")

    stab = Stabilizer(StabilizerParams())
    rows, Hs = [], []
    t0 = time.perf_counter()
    for idx, frame in frames(cap, args.width, start, stop):
        st = stab.update(frame)
        Hs.append(st.H)
        rows.append(dict(frame=idx, ok=st.ok, inliers=st.inliers, corrected=st.corrected,
                         correction_px=st.correction_px, keyframe=st.keyframe,
                         n_keyframes=st.n_keyframes,
                         **{f"h{i}{j}": st.H[i, j] for i in range(3) for j in range(3)}))
        if (idx - start) % 300 == 0:
            print(f"  quadro {idx}/{stop}  keyframes={st.n_keyframes}", flush=True)
    elapsed = time.perf_counter() - t0
    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stabilization.csv", index=False)
    corr = df.correction_px.dropna()
    # deslocamento da referência no quadro atual (quanto a câmera "andou")
    centers = np.array([np.linalg.inv(H) @ np.array([pw / 2, ph / 2, 1.0]) for H in Hs])
    centers = centers[:, :2] / centers[:, 2:]
    summary = {
        "video": info.path,
        "processing_size": [pw, ph],
        "frames": len(df),
        "frame_to_frame_ok_pct": round(100 * float(df.ok.mean()), 1),
        "median_inliers": int(df.inliers.median()),
        "corrections": int(df.corrected.sum()),
        "correction_px": {"median": round(float(corr.median()), 2) if len(corr) else None,
                          "p95": round(float(corr.quantile(.95)), 2) if len(corr) else None,
                          "max": round(float(corr.max()), 2) if len(corr) else None},
        "keyframes": int(df.n_keyframes.iloc[-1]),
        "camera_pan_range_px": {
            "x": [round(float(centers[:, 0].min())), round(float(centers[:, 0].max()))],
            "y": [round(float(centers[:, 1].min())), round(float(centers[:, 1].max()))]},
        "fps": round(len(df) / elapsed, 1),
    }
    (out_dir / "stabilization.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # segunda passada: mosaico e vídeo estabilizado
    T, size = canvas_from(Hs, pw, ph)
    cap = open_video(args.video)
    mosaic = np.zeros((size[1], size[0], 3), np.uint8)
    writer = None
    if args.save_video:
        vs = min(1.0, 1280 / size[0])
        vsize = (int(size[0] * vs) // 2 * 2, int(size[1] * vs) // 2 * 2)
        writer = cv2.VideoWriter(str(out_dir / "stabilized.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, vsize)
    corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
    for k, (idx, frame) in enumerate(frames(cap, args.width, start, stop)):
        M = T @ Hs[k]
        warped = cv2.warpPerspective(frame, M, size)
        mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, size) > 0
        if k % 15 == 0:
            mosaic[mask] = warped[mask]
        if writer is not None:
            view = (mosaic * 0.45).astype(np.uint8)
            view[mask] = warped[mask]
            poly = cv2.perspectiveTransform(corners, M).astype(np.int32)
            cv2.polylines(view, [poly], True, (0, 255, 255), 2)
            cv2.putText(view, f"{idx} {idx / info.fps:6.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
            writer.write(cv2.resize(view, vsize, interpolation=cv2.INTER_AREA))
    if writer is not None:
        writer.release()
    cv2.imwrite(str(out_dir / "mosaic.jpg"), mosaic)
    print(f"saídas em {out_dir}")


if __name__ == "__main__":
    main()
