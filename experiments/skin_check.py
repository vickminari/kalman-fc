"""Falsos positivos de pele (pés e mãos sob os refletores): a cor separa da bola? (Tarefa 4)

Hipótese ([010]): sob luz forte, a pele clara tem saturação baixa e brilho alto, parecida
com o branco da bola (S ≈ 18, V ≈ 216 a 640 px). Mas a pele ainda tem matiz de
vermelho/laranja (H perto de 0–25 no OpenCV) com saturação maior que a da bola.

Lê `outputs/experiments/<vídeo>/candidates.csv` (de `experiments/candidate_features.py`,
detector HSV, com rótulo bola = 1), relê os quadros marcados do vídeo e mede, para cada
candidato:

- mediana de H, S e V no disco do candidato;
- `skin`: fração de pixels "cor de pele" no disco: H ∈ [0, 25] ∪ [165, 180), S ≥ `skin_s_min`
  e V ≥ 60;
- `skin_ring`: a mesma fração num anel em volta (1,3 r a 2,5 r): pé ou mão perto da bola;
- `score_comb`: o score combinado do modelo salvo (`data/models/ball_scorer_<vídeo>.json`), e se
  o candidato passa no limiar (são esses os falsos que competem com a bola na trilha).

Saídas em `outputs/experiments/<vídeo>/`: `skin_stats.csv`, `skin_summary.json` (AUC de cada
medida, bola × falso, em todos e só nos que passam no score) e `skin_mosaic.png` (recortes dos
falsos com maior score e de algumas bolas, com a fração de pele).

Uso (a partir da raiz do repositório):

    python -m experiments.skin_check videos/campo_lateral.MOV
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detector import cool_fraction
from src.scoring import BallScorer
from src.video import frames, open_video, processing_size, video_info

SKIN_H = ((0, 25), (165, 180))   # matiz da pele (OpenCV, H em 0..180)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(atributo de uma bola > atributo de um falso), com empates valendo meio."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def skin_mask(hsv: np.ndarray, s_min: int = 25, v_min: int = 60) -> np.ndarray:
    """Pixels com matiz de pele, saturação ≥ s_min e brilho ≥ v_min (bool)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue = np.zeros(h.shape, bool)
    for lo, hi in SKIN_H:
        hue |= (h >= lo) & (h <= hi)
    return hue & (s >= s_min) & (v >= v_min)


def color_stats(hsv: np.ndarray, x: float, y: float, diameter: float, s_min: int = 25) -> dict:
    """Estatísticas de cor no disco do candidato e no anel em volta."""
    r = max(diameter / 2.0, 2.0)
    R = int(np.ceil(2.5 * r)) + 1
    xi, yi = int(round(x)), int(round(y))
    H, W = hsv.shape[:2]
    x0, x1, y0, y1 = max(xi - R, 0), min(xi + R + 1, W), max(yi - R, 0), min(yi + R + 1, H)
    crop = hsv[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    disc = d <= r
    ring = (d >= 1.3 * r) & (d <= 2.5 * r)
    sk = skin_mask(crop, s_min)
    px = crop[disc]
    return {"h_med": float(np.median(px[:, 0])) if len(px) else np.nan,
            "s_med": float(np.median(px[:, 1])) if len(px) else np.nan,
            "v_med": float(np.median(px[:, 2])) if len(px) else np.nan,
            "skin": float(sk[disc].mean()) if disc.any() else np.nan,
            "skin_ring": float(sk[ring].mean()) if ring.any() else np.nan,
            "cool": cool_fraction(hsv, x, y, r)}       # a mesma função do detector


def combined_z(df: pd.DataFrame, scorer: BallScorer) -> np.ndarray:
    """Score combinado do modelo salvo, a partir das colunas do candidates.csv."""
    X = df[scorer.features].copy()
    for j, f in enumerate(scorer.features):
        X[f] = X[f].fillna(np.nan)
    X["motion"] = np.log1p(X["motion"]) if "motion" in X else 0
    Xv = X.to_numpy(float)
    fill = np.asarray(scorer.fill)
    nan = np.isnan(Xv)
    Xv[nan] = np.broadcast_to(fill, Xv.shape)[nan]
    return scorer.z(Xv)


BASE_FEATURES = ["score", "circularity", "whiteness", "ring_grass", "motion", "lss_sim", "size_fit"]


def refit(df: pd.DataFrame, extras: dict[str, list[str]] | None = None) -> dict:
    """Reajusta o score combinado (regressão logística, mesma divisão treino/teste do
    candidate_features) com atributos de cor a mais, e compara no TESTE:
    AUC do score, "melhor candidato = bola" e falsos por quadro no limiar que mantém 95%
    das bolas do treino."""
    from experiments.candidate_features import logistic_fit
    extras = extras or {"base": [], "+skin": ["skin"], "+skin+skin_ring": ["skin", "skin_ring"],
                        "+cool": ["cool"], "+skin+cool": ["skin", "cool"]}
    test_frames = df[~df.train].frame.unique()
    n_test = len(test_frames)
    vis_test = df[~df.train & (df.label == 1)].frame.nunique()
    res = {}
    for name, extra in extras.items():
        feats = BASE_FEATURES + extra
        X = df[feats].copy()
        X = X.fillna(X[df.train].median())
        X["motion"] = np.log1p(X.motion)
        mu, sd = X[df.train].mean(), X[df.train].std().replace(0, 1)
        Z = ((X - mu) / sd).to_numpy()
        tr = df.train.to_numpy()
        w = logistic_fit(Z[tr], df.label.to_numpy()[tr].astype(float))
        comb = np.hstack([np.ones((len(Z), 1)), Z]) @ w
        g = df.assign(comb=comb)
        tst = g[~g.train]
        best = tst.loc[tst.groupby("frame").comb.idxmax()]
        thr = np.quantile(g[g.train & (g.label == 1)].comb, 0.05)
        keep = tst[tst.comb >= thr]
        res[name] = {
            "auc_test": round(auc(tst[tst.label == 1].comb, tst[tst.label == 0].comb), 4),
            "best_is_ball_test_pct": round(100 * int(best.label.sum()) / max(vis_test, 1), 1),
            "recall_test_pct": round(100 * keep[keep.label == 1].frame.nunique() / max(vis_test, 1), 1),
            "false_per_frame_test": round(float((keep.label == 0).sum()) / max(n_test, 1), 3),
            "weights": {k: round(float(v), 2) for k, v in zip(["bias"] + feats, w)},
        }
    return res


def mosaic(frames_bgr: dict, rows: pd.DataFrame, out: Path, title_col: str = "skin",
           n_cols: int = 10, size: int = 36, zoom: int = 3) -> None:
    tiles = []
    for _, r in rows.iterrows():
        img = frames_bgr.get(int(r.frame))
        if img is None:
            continue
        H, W = img.shape[:2]
        half = size // 2
        xi, yi = int(round(r.x)), int(round(r.y))
        pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)
        tile = pad[yi:yi + size, xi:xi + size]
        tile = cv2.resize(tile, (size * zoom, size * zoom), interpolation=cv2.INTER_NEAREST)
        col = (0, 255, 0) if r.label == 1 else (0, 0, 255)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), col, 2)
        cv2.putText(tile, f"{100 * r[title_col]:.0f}% S{r.s_med:.0f}", (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, f"q{int(r.frame)}", (3, tile.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % n_cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + n_cols]) for i in range(0, len(tiles), n_cols)])
    cv2.imwrite(str(out), grid)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--skin-s-min", type=int, default=25)
    args = ap.parse_args()

    stem = Path(args.video).stem
    exp_dir = Path("outputs/experiments") / stem
    df = pd.read_csv(exp_dir / "candidates.csv")
    df = df[df.detector == "hsv"].reset_index(drop=True)
    scorer = BallScorer.load(args.scorer or f"data/models/ball_scorer_{stem}.json")
    df["score_comb"] = combined_z(df, scorer)
    df["passes"] = df.score_comb >= scorer.threshold

    need = set(df.frame.unique())
    cap = open_video(args.video)
    info = video_info(cap, args.video)
    stats, keep_imgs = [], {}
    by_frame = {f: g for f, g in df.groupby("frame")}
    top_false = set(df[(df.label == 0) & df.passes].nlargest(60, "score_comb").frame)
    some_balls = set(df[df.label == 1].sample(min(20, int((df.label == 1).sum())), random_state=0).frame)
    for idx, frame in frames(cap, args.width, 0, max(need) + 1):
        if idx not in need:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i, r in by_frame[idx].iterrows():
            stats.append(dict(index=i, **color_stats(hsv, r.x, r.y, r.diameter, args.skin_s_min)))
        if idx in top_false or idx in some_balls:
            keep_imgs[idx] = frame
    cap.release()
    st = pd.DataFrame(stats).set_index("index")
    df = df.join(st)
    df.to_csv(exp_dir / "skin_stats.csv", index=False)

    cols = ["h_med", "s_med", "v_med", "skin", "skin_ring", "cool"]
    out = {"video": args.video, "n": int(len(df)), "balls": int(df.label.sum()),
           "passes_false": int(((df.label == 0) & df.passes).sum()),
           "passes_balls": int(((df.label == 1) & df.passes).sum())}
    for part, sel in (("all", np.ones(len(df), bool)), ("passes_score", df.passes.to_numpy())):
        g = df[sel]
        out[part] = {
            "auc_ball_vs_false (alto = bola)": {c: round(auc(g[g.label == 1][c], g[g.label == 0][c]), 3) for c in cols},
            "median_ball": {c: round(float(g[g.label == 1][c].median()), 3) for c in cols},
            "median_false": {c: round(float(g[g.label == 0][c].median()), 3) for c in cols},
            "frac_skin>0.2_ball": round(float((g[g.label == 1].skin > 0.2).mean()), 3),
            "frac_skin>0.2_false": round(float((g[g.label == 0].skin > 0.2).mean()), 3),
        }
    out["refit"] = refit(df)
    (exp_dir / "skin_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(pd.DataFrame(out["refit"]).T.to_string())

    sel = pd.concat([df[(df.label == 0) & df.passes].nlargest(40, "score_comb"),
                     df[(df.label == 1) & df.frame.isin(keep_imgs)].head(10)])
    mosaic(keep_imgs, sel, exp_dir / "skin_mosaic.png")
    print(f"mosaico em {exp_dir / 'skin_mosaic.png'} (vermelho = falso, verde = bola)")


if __name__ == "__main__":
    main()
