"""Repete só a TRILHA sobre candidatos guardados, para varrer parâmetros do filtro depressa.

O caro do pipeline é ler o vídeo, estabilizar, detectar e calcular o score combinado. Isso
não depende dos parâmetros do Kalman. Este experimento faz essa parte uma vez (cache em
`outputs/experiments/<vídeo>/replay_cache.pkl`) e depois roda o `BallTracker` para cada
combinação de parâmetros, com a MESMA lógica do `src/main.py` (F1 em px da referência ou
F2 em metros, com Σm propagado pelo jacobiano). Cada combinação passa pela régua de
`experiments/track_diagnostics.py` (report 011).

Com os parâmetros padrão, o resultado é o mesmo do `src.main` (conferido no report 013).

Uso (a partir da raiz do repositório):

    python -m experiments.replay_tracker videos/campo_lateral.MOV --metric --max-speed 1e9 20 30 40 --max-dark 10 15 20
    python -m experiments.replay_tracker videos/campo_centro.MOV --sigma-a 300 1000 4000
    python -m experiments.replay_tracker videos/campo_lateral.MOV --metric --max-speed 30 --save-logs --tag lateral_F2_v30

Opções em lista viram uma grade (todas as combinações). Resultados em
`outputs/experiments/<vídeo>/replay_<F1|F2>.csv` (ou `--out`).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.track_diagnostics import _row, diagnose, typical_Sm
from src.detector import BallDetector, DetectorParams
from src.evaluation import evaluate_track, load_annotations
from src.field import Calibration, FieldModel
from src.lss import to_lab
from src.motion_cue import MotionCue
from src.scoring import BallScorer
from src.stabilize import Stabilizer, StabilizerParams
from src.tracker import BallTracker, TrackerParams
from src.video import frames, open_video, processing_size, video_info


# ---------------------------------------------------------------------- cache dos candidatos
def collect(video: str, scorer_path: str, width: int = 640, cache: Path | None = None) -> dict:
    """Estabilização + detector + score combinado em todos os quadros (como no src.main).

    Guarda, por quadro: H (atual -> referência), candidatos que passaram no limiar do score
    (px do quadro atual) e a probabilidade de cada um ser a bola.
    """
    # o cache vale para este modelo de score (conteúdo do arquivo) e esta resolução
    scorer_hash = hashlib.sha1(Path(scorer_path).read_bytes()).hexdigest()
    if cache is not None and cache.exists():
        data = pickle.loads(cache.read_bytes())
        if data.get("scorer_hash") == scorer_hash and data.get("width") == width:
            return data
        print("  cache desatualizado (modelo do score ou resolução mudou): coletando de novo")
    cap = open_video(video)
    info = video_info(cap, video)
    pw, ph = processing_size(info.width, info.height, width)
    detector, stab = BallDetector(DetectorParams()), Stabilizer(StabilizerParams())
    scorer = BallScorer.load(scorer_path)
    motion = MotionCue()
    out = dict(video=video, scorer=scorer_path, scorer_hash=scorer_hash, width=width, fps=info.fps, size=(pw, ph),
               orig_width=info.width, frame=[], H=[], cands=[], scores=[])
    for idx, frame in frames(cap, width, 0, info.n_frames):
        H = stab.update(frame).H
        cands = detector.detect(frame).candidates
        cur = np.array([[c.x, c.y] for c in cands]).reshape(-1, 2)
        sc = np.zeros(0)
        motion.update(frame, H)
        if len(cands):
            mv = motion.scores(cur[:, 0], cur[:, 1], np.array([c.diameter for c in cands]))
            lab = to_lab(frame) if "lss_sim" in scorer.features else None
            z = scorer.z(scorer.feature_matrix(cands, mv, lab, pw))
            keep = z >= scorer.threshold
            cur, sc = cur[keep], BallScorer.prob(z[keep])
        out["frame"].append(idx)
        out["H"].append(H)
        out["cands"].append(cur)
        out["scores"].append(sc)
        if idx % 600 == 0:
            print(f"  coletando quadro {idx}/{info.n_frames}", flush=True)
    cap.release()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(out))
    return out


# ---------------------------------------------------------------------- trilha
def replay(data: dict, tparams: TrackerParams, field: FieldModel | None, metric: bool,
           sigma_m_px: float = 1.5, steps_out: dict | None = None) -> pd.DataFrame:
    """Roda a trilha sobre os candidatos guardados; devolve o log no formato do src.main.

    Com `steps_out` (um dict), guarda também o `TrackStep` de cada quadro (steps_out[quadro]),
    para quem precisa da região de busca (S, já com o teto) ou da predição, como o E3.
    """
    tracker = BallTracker(tparams)
    Sm_px = np.eye(2) * sigma_m_px ** 2
    fps = data["fps"]
    rows = []
    for idx, H, cur, sc in zip(data["frame"], data["H"], data["cands"], data["scores"]):
        cand_ref = Stabilizer.apply(H, cur) if len(cur) else cur
        Sm = None
        if metric:
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
        step = tracker.step(cand_trk, sc, Sm)
        if steps_out is not None:
            steps_out[idx] = step
        Hinv = np.linalg.inv(H)
        row = dict(frame=idx, t=idx / fps, state=step.state.value, n_cand=step.n_candidates,
                   dark=step.dark, new_track=step.new_track, lost=step.lost,
                   z_x=np.nan, z_y=np.nan, est_x=np.nan, est_y=np.nan)
        if step.x is not None:
            ref = field.to_image(step.x[:2]) if metric else step.x[:2].reshape(1, 2)
            row["est_x"], row["est_y"] = Stabilizer.apply(Hinv, ref)[0]
            row.update(px=step.x[0], py=step.x[1], vx=step.x[2], vy=step.x[3])
            if field is not None:
                if metric:
                    row["speed_kmh"] = float(np.hypot(step.x[2], step.x[3]) * 3.6)
                else:
                    a, b = field.to_world(np.array([step.x[:2], step.x[:2] + step.x[2:4] / fps]))
                    row["speed_kmh"] = float(np.linalg.norm(b - a) * fps * 3.6)
        if step.z_index is not None:
            row["z_x"], row["z_y"] = cur[step.z_index]
        row.update({f"h{i}{j}": H[i, j] for i in range(3) for j in range(3)})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- linha de comando
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--scorer", default=None, help="padrão: data/models/ball_scorer_<vídeo>.json")
    ap.add_argument("--metric", action="store_true", help="filtro F2 (metros)")
    ap.add_argument("--sigma-a", type=float, nargs="+", default=[None])
    ap.add_argument("--max-speed", type=float, nargs="+", default=[None],
                    help="px/s (F1) ou m/s (F2); 1e9 = desligado")
    ap.add_argument("--max-dark", type=int, nargs="+", default=[None])
    ap.add_argument("--sigma-m", type=float, default=TrackerParams.sigma_m, help="px")
    ap.add_argument("--jump", type=float, default=20.0)
    ap.add_argument("--save-logs", action="store_true", help="salva track.csv/track.json de cada combinação")
    ap.add_argument("--tag", default=None, help="prefixo das pastas com --save-logs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stem = Path(args.video).stem
    scorer = args.scorer or f"data/models/ball_scorer_{stem}.json"
    exp_dir = Path("outputs/experiments") / stem
    data = collect(args.video, scorer, cache=exp_dir / "replay_cache.pkl")
    pw, ph = data["size"]
    ann = load_annotations(Path("data/annotations") / f"{stem}.csv", scale=pw / data["orig_width"])
    field_path = Path("data/calib") / f"{stem}_field.json"
    field = FieldModel(Calibration.load(field_path), pw) if field_path.exists() else None
    if args.metric and field is None:
        raise SystemExit("--metric precisa da calibração do campo")

    results, table = [], []
    for sa, vmax, md in itertools.product(args.sigma_a, args.max_speed, args.max_dark):
        kw = {}
        if sa is not None:
            kw["sigma_a"] = sa
        if vmax is not None:
            kw["max_speed"] = vmax
        if md is not None:
            kw["max_dark"] = md
        if args.metric:
            tp = TrackerParams.metric(fps=data["fps"], **kw)
        else:
            tp = TrackerParams(fps=data["fps"], sigma_m=args.sigma_m, **kw)
        log = replay(data, tp, field, args.metric, args.sigma_m)
        Sm = typical_Sm(log, args.sigma_m, field if args.metric else None)
        r = diagnose(log, ann, data["fps"], tp.sigma_a, Sm, jump_px=args.jump)
        name = f"{'F2' if args.metric else 'F1'}_sa{tp.sigma_a:g}_v{tp.max_speed:g}_d{tp.max_dark}"
        row = _row(name, r)
        row.update(sigma_a=tp.sigma_a, max_speed=tp.max_speed, max_dark=tp.max_dark,
                   recovered_10_30=r["dark_episodes"]["recovered_10_30"])
        table.append(row)
        results.append(dict(name=name, params=dict(sigma_a=tp.sigma_a, max_speed=tp.max_speed,
                                                   max_dark=tp.max_dark), diagnostics=r))
        print(f"  {name}: na bola (teste) {r['reference']['on_ball_test_pct']}%, "
              f"saltos {r['jumps']['total']}", flush=True)
        if args.save_logs:
            d = Path("outputs") / f"{args.tag or stem}_{name}"
            d.mkdir(parents=True, exist_ok=True)
            log.to_csv(d / "track.csv", index=False)
            meta = {"video": args.video, "processing_size": [pw, ph],
                    "params": {"filter": "F2 (metros)" if args.metric else "F1 (px)",
                               "sigma_a": tp.sigma_a, "sigma_m_px": args.sigma_m,
                               "max_dark": tp.max_dark, "max_speed": tp.max_speed,
                               "scorer": scorer, "field": str(field_path) if field else None},
                    "vs_annotations": evaluate_track(log, ann)}
            (d / "track.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    df = pd.DataFrame(table).set_index("tag")
    out = Path(args.out) if args.out else exp_dir / f"replay_{'F2' if args.metric else 'F1'}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    (out.with_suffix(".json")).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.drop(columns=["sigma_a", "max_speed", "max_dark"]).T.to_string())
    print(f"tabela em {out}")


if __name__ == "__main__":
    main()
