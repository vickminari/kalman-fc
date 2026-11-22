"""Kalman FC — pipeline completo: captura -> estabilização -> detecção -> Kalman -> vídeo.

Uso (a partir da raiz do repositório):

    python -m src.main videos/campo_lateral.MOV --scorer data/models/ball_scorer_campo_lateral.json --save-video
    python -m src.main videos/campo_lateral.MOV --metric ...     # filtro F2, em metros no campo
    python -m src.main videos/campo_centro.MOV --start 30 --end 60 --show
    python -m src.main 0 --show                                  # câmera ao vivo (índice 0)

Opções do filtro:  --sigma-a  --sigma-m  --gate (prob. χ²)  --max-dark (quadros)
    F1 (padrão): coordenadas em px do quadro de referência; σa em px/s², σm em px.
    F2 (--metric): coordenadas em METROS no campo; σa em m/s², e Σm de cada quadro vem da
        propagação de σm (px) pela homografia: Σm_campo = J Σm_px Jᵀ.

Campo: se existir data/calib/<vídeo>_field.json (tools/calibrate_field.py), o vídeo de saída
ganha o minimapa e a velocidade em km/h. --no-field desliga.

Saídas em outputs/<vídeo>/ (ou outputs/<--tag>/):
    track.csv       uma linha por quadro: estado, medição aceita, predição, estimativa, S, H, X/Y (m)
    track.json      parâmetros, desempenho e (se houver marcações) comparação com a verdade
    tracked.mp4     (com --save-video) vídeo com as sobreposições (+ minimapa)
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
from src.field import Calibration, FieldModel
from src.gating import chi2_threshold
from src.lss import to_lab
from src.minimap import Minimap
from src.motion_cue import MotionCue
from src.scoring import BallScorer
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
    ap.add_argument("--sigma-a", type=float, default=None, help="padrão: 4000 px/s² (F1) ou 150 m/s² (F2)")
    ap.add_argument("--sigma-m", type=float, default=TrackerParams.sigma_m, help="ruído da medição, px")
    ap.add_argument("--gate", type=float, default=TrackerParams.gate_prob)
    ap.add_argument("--max-dark", type=int, default=TrackerParams.max_dark)
    ap.add_argument("--no-stab", action="store_true", help="desliga a compensação da câmera")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--show", action="store_true", help="mostra a janela (q/ESC sai, espaço pausa)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default=None, help="nome da pasta de saída (padrão: nome do vídeo)")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--scorer", default=None,
                    help="modelo de score combinado (JSON de experiments/candidate_features.py)")
    ap.add_argument("--no-filter", action="store_true",
                    help="com --scorer: não descarta candidatos abaixo do limiar, só reordena")
    ap.add_argument("--field", default=None, help="calibração do campo (padrão: data/calib/<vídeo>_field.json)")
    ap.add_argument("--no-field", action="store_true")
    ap.add_argument("--metric", action="store_true", help="filtro F2: Kalman em metros no campo")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cap = open_video(args.video)
    info = video_info(cap, str(args.video))
    is_file = not str(args.video).isdigit()
    stem = Path(args.video).stem if is_file else f"camera{args.video}"
    pw, ph = processing_size(info.width, info.height, args.width)
    start = int(args.start * info.fps)
    stop = int(args.end * info.fps) if args.end else (info.n_frames if is_file else None)
    out_dir = Path(args.out) / (args.tag or stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- campo (homografia do chão) e modo do filtro
    field = None
    field_path = Path(args.field) if args.field else Path("data/calib") / f"{stem}_field.json"
    if not args.no_field and field_path.exists():
        field = FieldModel(Calibration.load(field_path), pw)
    if args.metric and field is None:
        raise SystemExit("--metric precisa da calibração do campo (tools/calibrate_field.py)")
    if args.no_stab and field is not None:
        print("aviso: sem estabilização, a homografia do campo só vale enquanto a câmera não se mexer")

    if args.metric:
        tparams = TrackerParams.metric(fps=info.fps, gate_prob=args.gate, max_dark=args.max_dark,
                                       **({"sigma_a": args.sigma_a} if args.sigma_a else {}))
    else:
        tparams = TrackerParams(fps=info.fps, sigma_a=args.sigma_a or TrackerParams.sigma_a,
                                sigma_m=args.sigma_m, gate_prob=args.gate, max_dark=args.max_dark)
    Sm_px = np.eye(2) * args.sigma_m ** 2

    detector = BallDetector(DetectorParams())
    stab = None if args.no_stab else Stabilizer(StabilizerParams())
    tracker = BallTracker(tparams)
    scorer = BallScorer.load(args.scorer) if args.scorer else None
    motion = MotionCue() if scorer is not None else None
    overlay = Overlay(gate_threshold=chi2_threshold(args.gate))
    minimap = Minimap(field, ph, (pw, ph)) if field is not None else None
    out_size = (pw + (minimap.w if minimap else 0), ph)
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(out_dir / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, out_size)

    # conversões entre coordenadas do filtro (F1: px da referência; F2: metros) e o resto
    def track_to_ref(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return field.to_image(p) if args.metric else p

    def track_to_world(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return p if args.metric else field.to_world(p)

    def world_speed_kmh(x):
        """Velocidade no campo (km/h) a partir do estado do filtro."""
        if field is None or x is None:
            return None
        if args.metric:
            return float(np.hypot(x[2], x[3]) * 3.6)
        dt = 1.0 / info.fps
        a, b = field.to_world(np.array([x[:2], x[:2] + x[2:4] * dt]))
        return float(np.linalg.norm(b - a) / dt * 3.6)

    print(f"{info.path}: {info.width}x{info.height} -> {pw}x{ph}, fps={info.fps:.2f}, "
          f"filtro {'F2 (metros)' if args.metric else 'F1 (px)'}, sigma_a={tparams.sigma_a}, "
          f"gate={args.gate}, max_dark={args.max_dark}"
          f"{', campo: ' + str(field_path) if field else ''}{' (sem estabilização)' if stab is None else ''}")
    if field is not None and start > 0:
        print(f"  (estabilizando os quadros 0..{start} antes: a calibração do campo é do quadro 0)")

    rows = []
    t_total = 0.0
    paused = False
    first = 0 if (field is not None and stab is not None) else start
    for idx, frame in frames(cap, args.width, first, stop):
        if idx < start:                      # só leva a estabilização até o quadro de início
            stab.update(frame)
            continue
        t0 = time.perf_counter()
        H = stab.update(frame).H if stab is not None else np.eye(3)
        det = detector.detect(frame)
        cands = det.candidates
        cand_cur = np.array([[c.x, c.y] for c in cands]).reshape(-1, 2)
        scores = np.array([c.score for c in cands])
        if scorer is not None:
            # score combinado: movimento (câmera compensada) + LSS + tamanho esperado + detector
            motion.update(frame, H)
            if len(cands):
                mv = motion.scores(cand_cur[:, 0], cand_cur[:, 1], np.array([c.diameter for c in cands]))
                lab = to_lab(frame) if "lss_sim" in scorer.features else None
                z = scorer.z(scorer.feature_matrix(cands, mv, lab, pw))
                keep = np.ones(len(z), bool) if args.no_filter else z >= scorer.threshold
                cand_cur, scores = cand_cur[keep], BallScorer.prob(z[keep])
        cand_ref = Stabilizer.apply(H, cand_cur) if len(cand_cur) else cand_cur

        Sm = None
        if args.metric:
            # F2: medições no campo; Σm propagado pelo jacobiano no ponto atual da trilha
            cand_trk = field.to_world(cand_ref) if len(cand_ref) else cand_ref
            if tracker.kf is not None:
                uv = field.to_image(tracker.kf.x[:2])[0]
            elif len(cand_ref):
                uv = cand_ref[0]
            else:
                uv = None
            if uv is not None:
                J = field.jacobian(uv)
                Sm = J @ Sm_px @ J.T
        else:
            cand_trk = cand_ref
        step = tracker.step(cand_trk, scores, Sm)
        step.frame = idx
        t_total += time.perf_counter() - t0

        Hinv = np.linalg.inv(H)
        est_cur = Stabilizer.apply(Hinv, track_to_ref(step.x[:2]))[0] if step.x is not None else (np.nan, np.nan)
        z_cur = cand_cur[step.z_index] if step.z_index is not None else (np.nan, np.nan)
        row = dict(frame=idx, t=idx / info.fps, state=step.state.value, n_cand=step.n_candidates,
                   z_x=z_cur[0], z_y=z_cur[1], est_x=est_cur[0], est_y=est_cur[1],
                   dark=step.dark, new_track=step.new_track, lost=step.lost)
        if step.z is not None:
            row.update(z_trk_x=step.z[0], z_trk_y=step.z[1])
        if step.x is not None:
            row.update(px=step.x[0], py=step.x[1], vx=step.x[2], vy=step.x[3])
        if step.z_pred is not None:
            row.update(pred_x=step.z_pred[0], pred_y=step.z_pred[1],
                       S_xx=step.S[0, 0], S_xy=step.S[0, 1], S_yy=step.S[1, 1])
        if step.d2 is not None and step.z_index is not None:
            row["d2"] = float(step.d2[step.z_index])
        est_w = meas_w = None
        speed = None
        if field is not None:
            if step.x is not None:
                est_w = track_to_world(step.x[:2])[0]
                speed = world_speed_kmh(step.x)
                row.update(world_X=est_w[0], world_Y=est_w[1], speed_kmh=speed)
            if step.z is not None:
                meas_w = track_to_world(step.z)[0]
                row.update(meas_X=meas_w[0], meas_Y=meas_w[1])
        row.update({f"h{i}{j}": H[i, j] for i in range(3) for j in range(3)})
        rows.append(row)

        if writer is not None or args.show:
            vis = overlay.draw(frame, step, H, cand_cur, info.fps,
                               to_ref=track_to_ref if args.metric else None,
                               speed_text=None if speed is None else f"v={speed:4.1f} km/h")
            if minimap is not None:
                mm = minimap.draw(est_w, meas_w, speed, new_track=step.new_track,
                                  dark=step.state.value == "NO_ESCURO")
                vis = np.hstack([vis, mm])
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
        "params": {"filter": "F2 (metros)" if args.metric else "F1 (px)",
                   "sigma_a": tparams.sigma_a, "sigma_m_px": args.sigma_m, "gate_prob": args.gate,
                   "max_dark": args.max_dark, "stabilization": stab is not None,
                   "scorer": args.scorer, "scorer_filter": scorer is not None and not args.no_filter,
                   "field": str(field_path) if field else None},
        "frames": n,
        "pipeline_fps": round(n / t_total, 1) if t_total else None,
        "states_pct": counts,
        "tracks_started": int(log.new_track.sum()),
        "tracks_lost": int(log.lost.sum()),
    }
    if field is not None and "speed_kmh" in log:
        sp = log.speed_kmh.dropna()
        summary["speed_kmh"] = {"median": round(float(sp.median()), 1), "p95": round(float(sp.quantile(.95)), 1)}
    ann_path = Path(args.annotations) if args.annotations else Path("data/annotations") / f"{stem}.csv"
    if is_file and ann_path.exists():
        ann = load_annotations(ann_path, scale=pw / info.width)
        summary["vs_annotations"] = evaluate_track(log, ann)
    (out_dir / "track.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"saidas em {out_dir}")


if __name__ == "__main__":
    main()
