"""Experimento: o que separa a bola dos candidatos falsos? E qual gerador de candidatos é melhor?

Para cada quadro MARCADO do vídeo:
- roda três geradores de candidatos: HSV (src/detector.py), LoG em escala e
  SimpleBlobDetector do OpenCV (src/blob_detector.py);
- calcula atributos de cada candidato: score do detector, circularidade, brancura, grama
  no anel, diâmetro, linha (y), MOVIMENTO com a câmera compensada (src/motion_cue.py) e
  o descritor de AUTO-SIMILARIDADE LOCAL (src/lss.py);
- rotula cada candidato: é a bola se estiver a ≤ `radius` px da marcação.

Análise, com divisão temporal (treino = primeira metade dos quadros marcados, teste =
segunda metade), para não avaliar no mesmo trecho em que algo foi ajustado:
1. Geradores: recall (bola entre os candidatos), falsos por quadro, "melhor = bola", ms/quadro.
2. Atributos (no HSV): AUC de cada atributo para separar bola × falso, no teste.
   AUC = probabilidade de um candidato-bola ter o atributo "mais de bola" que um falso
   (0,5 = inútil, 1 = perfeito). O molde do LSS e o prior de tamanho são ajustados no treino.
3. Score combinado: regressão logística (treino) sobre os atributos; no teste, "melhor = bola"
   com o score original × o combinado.

Uso:
    python -m experiments.candidate_features videos/campo_lateral.MOV
Saídas em outputs/experiments/<vídeo>/: candidates.csv, summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.blob_detector import LoGBlobDetector, OpenCVBlobDetector
from src.detector import BallDetector
from src.evaluation import load_annotations
from src.lss import LSSVerifier, descriptor, to_lab
from src.motion_cue import MotionCue
from src.scoring import BallScorer
from src.stabilize import Stabilizer
from src.video import frames, open_video, processing_size, video_info

FEATURES = ["score", "circularity", "whiteness", "ring_grass", "motion", "lss_sim", "size_fit", "cool"]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC por Mann-Whitney (empates contam meio)."""
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """Regressão logística por Newton (IRLS) com regularização L2. X já padronizado."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    R = np.eye(len(w)) * l2
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + R @ w
        Hs = Xb.T @ (Xb * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--out", default="outputs/experiments")
    args = ap.parse_args()

    cap = open_video(args.video)
    info = video_info(cap, args.video)
    pw, _ = processing_size(info.width, info.height, args.width)
    stem = Path(args.video).stem
    ann_path = Path(args.annotations) if args.annotations else Path("data/annotations") / f"{stem}.csv"
    ann = load_annotations(ann_path, scale=pw / info.width).set_index("frame")
    marked = set(ann.index)
    last = max(marked)
    out_dir = Path(args.out) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    detectors = {"hsv": BallDetector(), "log": LoGBlobDetector(), "cvblob": OpenCVBlobDetector()}
    times = {k: 0.0 for k in detectors}
    stab, motion = Stabilizer(), MotionCue(gap=3, search=2)
    rows, lss_rows = [], []
    t_lss = 0.0
    n_marked = 0
    for idx, frame in frames(cap, args.width, 0, last + 1):
        H = stab.update(frame).H
        motion.update(frame, H)
        if idx not in marked:
            continue
        n_marked += 1
        a = ann.loc[idx]
        lab = to_lab(frame)
        for name, det in detectors.items():
            t0 = time.perf_counter()
            d = det.detect(frame)
            times[name] += time.perf_counter() - t0
            if not d.candidates:
                continue
            xs = np.array([c.x for c in d.candidates])
            ys = np.array([c.y for c in d.candidates])
            ds = np.array([c.diameter for c in d.candidates])
            mv = motion.scores(xs, ys, ds)
            for j, c in enumerate(d.candidates):
                dist = np.hypot(c.x - a.gx, c.y - a.gy) if a.visible == 1 else np.inf
                rows.append(dict(frame=idx, detector=name, x=c.x, y=c.y, diameter=c.diameter,
                                 circularity=c.circularity, whiteness=c.whiteness,
                                 ring_grass=c.ring_grass, cool=c.cool, score=c.score, motion=mv[j],
                                 dist=dist, label=int(dist <= args.radius)))
                t0 = time.perf_counter()
                desc = descriptor(lab, c.x, c.y, c.diameter)
                t_lss += time.perf_counter() - t0
                lss_rows.append(desc if desc is not None else np.full(36, np.nan))
        if n_marked % 100 == 0:
            print(f"  {n_marked}/{len(marked)} quadros marcados processados", flush=True)
    cap.release()

    df = pd.DataFrame(rows)
    L = np.array(lss_rows)
    vis_frames = set(ann.index[ann.visible == 1])
    split = int(np.median(sorted(marked)))
    df["train"] = df.frame < split

    # --- LSS: molde = média das bolas verdadeiras do treino (detector HSV)
    tr_pos = (df.detector == "hsv") & df.train & (df.label == 1)
    ver = LSSVerifier().fit(L[tr_pos.values])
    df["lss_dist"] = [ver.distance(l if not np.isnan(l).any() else None) for l in L]
    df["lss_sim"] = -df.lss_dist
    # --- prior de tamanho: log(d) = a + b·y ajustado nas bolas do treino
    pos = df[tr_pos]
    b, a0 = np.polyfit(pos.y, np.log(pos.diameter), 1)
    df["size_fit"] = -np.abs(np.log(df.diameter) - (a0 + b * df.y))
    df.drop(columns=[]).to_csv(out_dir / "candidates.csv", index=False)

    summary: dict = {"video": args.video, "annotated_frames": len(marked),
                     "ball_visible_frames": len(vis_frames), "split_frame": split,
                     "radius_px": args.radius, "lss_ms_per_candidate": round(1000 * t_lss / max(len(df), 1), 3),
                     "size_prior": {"log_d = a + b*y": [round(a0, 4), round(b, 5)]}}

    # --- 1. geradores
    gens = {}
    for name in detectors:
        g = df[df.detector == name]
        out = {}
        for part, sel in (("all", slice(None)), ("test", g.frame >= split)):
            gg = g[sel] if part == "test" else g
            vf = [f for f in vis_frames if (part == "all" or f >= split)]
            nf = len([f for f in marked if (part == "all" or f >= split)])
            hit = gg[gg.label == 1].frame.nunique()
            best = gg.loc[gg.groupby("frame").score.idxmax()]
            out[part] = {
                "recall_pct": round(100 * hit / max(len(vf), 1), 1),
                "best_is_ball_pct": round(100 * int(best.label.sum()) / max(len(vf), 1), 1),
                "false_per_frame": round(float((gg.label == 0).sum()) / max(nf, 1), 2),
            }
        out["ms_per_frame"] = round(1000 * times[name] / max(n_marked, 1), 2)
        gens[name] = out
    summary["generators"] = gens

    # --- 2. AUC dos atributos (HSV, teste)
    h = df[df.detector == "hsv"]
    te = h[~h.train]
    summary["feature_auc_test_hsv"] = {f: round(auc(te[te.label == 1][f].values, te[te.label == 0][f].values), 3)
                                       for f in FEATURES}
    summary["feature_auc_test_log"] = {f: round(auc(*(df[(df.detector == "log") & ~df.train & (df.label == k)][f].values
                                                      for k in (1, 0))), 3) for f in FEATURES}

    # --- 3. score combinado (logística), por detector
    comb = {}
    for name in ("hsv", "log"):
        g = df[df.detector == name].copy()
        feats = FEATURES   # "cool" (cor da bola) entrou em 22/11, report 015
        tr = g[g.train]
        med = tr[feats].median()
        X = g[feats].fillna(med)
        X["motion"] = np.log1p(X.motion)
        mu, sd = X[g.train].mean(), X[g.train].std().replace(0, 1)
        Z = ((X - mu) / sd).values
        w = logistic_fit(Z[g.train.values], g.label.values[g.train.values].astype(float))
        g["comb"] = np.hstack([np.ones((len(Z), 1)), Z]) @ w
        res = {"weights": {k: round(float(v), 3) for k, v in zip(["bias"] + feats, w)}}
        tst = g[~g.train]
        vf_test = [f for f in vis_frames if f >= split]
        for col in ("score", "comb"):
            best = tst.loc[tst.groupby("frame")[col].idxmax()]
            res[f"best_is_ball_test_pct[{col}]"] = round(100 * int(best.label.sum()) / max(len(vf_test), 1), 1)
        res["auc_test[comb]"] = round(auc(tst[tst.label == 1].comb.values, tst[tst.label == 0].comb.values), 3)
        # quantos falsos sobram se cortarmos no limiar que mantém 95% das bolas do treino
        thr = np.quantile(g[g.train & (g.label == 1)].comb, 0.05)
        if name == "hsv":
            fill = med.copy()
            fill["motion"] = np.log1p(fill["motion"])
            model = BallScorer(features=feats, mean=[float(v) for v in mu], std=[float(v) for v in sd],
                               weights=[float(v) for v in w], fill=[float(fill[f]) for f in feats],
                               threshold=float(thr), size_prior=[float(a0), float(b)],
                               lss_template=[float(v) for v in ver.template],
                               reference_width=args.width,
                               trained_on=f"{stem}: quadros marcados < {split}")
            model_path = Path("data/models") / f"ball_scorer_{stem}.json"
            model.save(model_path)
            res["model"] = str(model_path)
        keep = tst[tst.comb >= thr]
        res["test_at_95pct_train_recall"] = {
            "recall_pct": round(100 * keep[keep.label == 1].frame.nunique() / max(len(vf_test), 1), 1),
            "false_per_frame": round(float((keep.label == 0).sum()) / max(len([f for f in marked if f >= split]), 1), 2)}
        comb[name] = res
    summary["combined"] = comb

    summary["ablation_hsv"] = ablation(df[df.detector == "hsv"].copy(), marked, vis_frames, split)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def ablation(g: pd.DataFrame, marked: set, vis_frames: set, split: int) -> dict:
    """Score combinado com subconjuntos de atributos: quanto cada pista contribui (teste)."""
    base = ["score", "circularity", "whiteness", "ring_grass"]
    sets = {
        "base (atributos do detector)": base,
        "base + movimento": base + ["motion"],
        "base + LSS": base + ["lss_sim"],
        "base + tamanho": base + ["size_fit"],
        "base + movimento + tamanho": base + ["motion", "size_fit"],
        "base + LSS + tamanho": base + ["lss_sim", "size_fit"],
        "base + movimento + LSS": base + ["motion", "lss_sim"],
        "todos sem cor": base + ["motion", "lss_sim", "size_fit"],
        "base + cor": base + ["cool"],
        "todos": base + ["motion", "lss_sim", "size_fit", "cool"],
    }
    vf_test = [f for f in vis_frames if f >= split]
    nf_test = len([f for f in marked if f >= split])
    out = {}
    for name, feats in sets.items():
        tr = g[g.train]
        X = g[feats].fillna(tr[feats].median())
        if "motion" in X:
            X["motion"] = np.log1p(X.motion)
        mu, sd = X[g.train].mean(), X[g.train].std().replace(0, 1)
        Z = ((X - mu) / sd).values
        w = logistic_fit(Z[g.train.values], g.label.values[g.train.values].astype(float))
        g["c"] = np.hstack([np.ones((len(Z), 1)), Z]) @ w
        tst = g[~g.train]
        best = tst.loc[tst.groupby("frame").c.idxmax()]
        thr = np.quantile(g[g.train & (g.label == 1)].c, 0.05)
        keep = tst[tst.c >= thr]
        out[name] = {
            "auc": round(auc(tst[tst.label == 1].c.values, tst[tst.label == 0].c.values), 3),
            "best_is_ball_pct": round(100 * int(best.label.sum()) / max(len(vf_test), 1), 1),
            "recall_at_thr_pct": round(100 * keep[keep.label == 1].frame.nunique() / max(len(vf_test), 1), 1),
            "false_per_frame_at_thr": round(float((keep.label == 0).sum()) / max(nf_test, 1), 2),
        }
    return out


if __name__ == "__main__":
    main()
