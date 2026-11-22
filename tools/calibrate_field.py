"""Calibra o campo de um vídeo a partir dos marcos (data/calib/<vídeo>_landmarks.json).

Uso (a partir da raiz do repositório):

    python -m tools.calibrate_field videos/campo_lateral.MOV
    python -m tools.calibrate_field videos/campo_centro.MOV

Lê as opções do bloco "calibration" do arquivo de marcos:
    fit_f: true            estima a distância focal junto com a pose
    f: 1743.2              usa esta distância focal (px em 1920 de largura)
    priors: {...}          conhecimento aproximado (ver src/field.calibrate)
Sem nenhuma delas, usa a K derivada da calibração de Zhang das fotos (iphone15_video_K).

Saídas:
    data/calib/<vídeo>_field.json      calibração (K, R, t, homografias, resíduos)
    outputs/<vídeo>/field_overlay.jpg  campo projetado sobre o quadro de calibração
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.field import POSTS_X, GOAL_LINE_LENGTH, LandmarkSet, calibrate, iphone15_video_K


def draw_overlay(img: np.ndarray, calib, lm: LandmarkSet, depth: float = 15.0) -> np.ndarray:
    out = img.copy()

    def seg(P, Q, col, th=2):
        pts = np.linspace(P, Q, 80)
        uv = calib.project(pts)
        depth = (calib.R @ pts.T).T[:, 2] + calib.t[2]          # só o que está na frente da câmera
        ok = np.all(np.abs(uv) < 1e5, axis=1) & (depth > 0.3)
        if ok.sum() >= 2:
            cv2.polylines(out, [uv[ok].astype(np.int32)], False, col, th, cv2.LINE_AA)

    for X in range(0, int(GOAL_LINE_LENGTH) + 1):
        seg([X, 0, 0], [X, depth, 0], (0, 200, 255), 1)
    for Y in range(0, int(depth) + 1):
        seg([0, Y, 0], [GOAL_LINE_LENGTH, Y, 0], (0, 200, 255), 1)
    seg([0, 0, 0], [GOAL_LINE_LENGTH, 0, 0], (0, 0, 255), 3)                 # linha do gol
    seg([0, 0, 0], [0, depth, 0], (255, 0, 0), 3)                            # laterais
    seg([GOAL_LINE_LENGTH, 0, 0], [GOAL_LINE_LENGTH, depth, 0], (255, 0, 0), 3)
    h = calib.shared.get("h", 2.0)
    for X in POSTS_X:
        seg([X, 0, 0], [X, 0, h], (255, 0, 255), 3)
    seg([POSTS_X[0], 0, h], [POSTS_X[1], 0, h], (255, 0, 255), 3)
    for p in lm.points:
        cv2.circle(out, tuple(int(round(v)) for v in p.xy), 7, (0, 255, 0), 2)
    cv2.putText(out, f"f={calib.K[0, 0]:.0f}px  rms={calib.rms_px:.2f}px  h={h:.2f}m  "
                     f"camera=({calib.camera_center[0]:.1f}, {calib.camera_center[1]:.1f}, "
                     f"{calib.camera_center[2]:.2f}) m",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--landmarks", default=None)
    args = ap.parse_args()

    stem = Path(args.video).stem
    lm_path = Path(args.landmarks) if args.landmarks else Path("data/calib") / f"{stem}_landmarks.json"
    lm = LandmarkSet.load(lm_path)
    opts = json.loads(lm_path.read_text(encoding="utf-8")).get("calibration", {})
    K = iphone15_video_K(*lm.image_size)
    if "f" in opts:
        K[0, 0] = K[1, 1] = float(opts["f"])
    priors = {k: tuple(v) for k, v in opts.get("priors", {}).items()}
    calib = calibrate(lm, K, fit_f=bool(opts.get("fit_f", False)), priors=priors)

    out_json = Path("data/calib") / f"{stem}_field.json"
    calib.save(out_json)
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lm.frame)
    ok, img = cap.read()
    if ok:
        out_dir = Path("outputs") / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "field_overlay.jpg"), draw_overlay(img, calib, lm))
    print(f"f = {calib.K[0, 0]:.1f} px   rms = {calib.rms_px:.2f} px   "
          f"incógnitas = { {k: round(v, 3) for k, v in calib.shared.items()} }")
    print(f"câmera em X={calib.camera_center[0]:.2f} Y={calib.camera_center[1]:.2f} "
          f"Z={calib.camera_center[2]:.2f} m")
    print("resíduos (px):", calib.residuals_px)
    print(f"salvo em {out_json}")


if __name__ == "__main__":
    main()
