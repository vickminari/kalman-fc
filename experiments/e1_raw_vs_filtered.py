"""E1 — trajetória bruta do detector × trajetória filtrada (experimento obrigatório).

Compara três trajetórias no mesmo vídeo, contra as marcações à mão:

- **detector bruto:** em cada quadro, o candidato de MAIOR score combinado (sem filtro e sem
  gating: é o que se teria usando só o detector);
- **medição aceita:** o candidato que o gating do filtro aceitou (a medição crua que entra
  na correção; não existe nos quadros no escuro);
- **estimativa filtrada:** o x̂ do Kalman (inclui a predição nos quadros no escuro).

Medidas (px a 640; erros no quadro atual, como `src/evaluation.py`; saltos, tremor e σ local
no quadro de referência, estabilizado):

- **cobertura** e **"na bola"** (a ≤ 10 px da marcação), sobre os quadros marcados com a bola visível;
- **erro médio e mediano** contra a marcação, nos quadros em que a trajetória está na bola;
- **saídas com a bola invisível** (quadros marcados como bola invisível em que o detector
  devolve um ponto, ou o filtro aceita uma medição): falsos positivos;
- **saltos** > 20 px entre quadros consecutivos (troca de alvo);
- **tremor** (norma da 2ª diferença, mediana) e **σ local** (resíduo de uma parábola em
  janelas de 7 quadros, como em `experiments/sigma_m.py`): a "variância da posição" do
  enunciado. Medição e estimativa são medidas nos MESMOS quadros (trechos com medição);
- **atraso** (quadros): o deslocamento s que minimiza o erro mediano entre a trajetória no
  quadro f + s e a marcação no quadro f (ver `experiments/e2_sigma_a.lag_frames`).

Gráfico (`outputs/experiments/<vídeo>/e1_<F1|F2>_trecho_<quadro>.png`): as três trajetórias
no trecho em que o detector bruto mais deixa de estar na bola (sem candidato ou candidato
errado) enquanto a estimativa fica nela.

Uso (a partir da raiz do repositório):

    python -m experiments.e1_raw_vs_filtered videos/campo_lateral.MOV
    python -m experiments.e1_raw_vs_filtered videos/campo_lateral.MOV --metric
    python -m experiments.e1_raw_vs_filtered videos/campo_centro.MOV --segment 2400 2490
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e2_sigma_a import annotations_ref, lag_frames
from experiments.replay_tracker import collect, replay
from experiments.sigma_m import moving_sigma
from experiments.track_diagnostics import H_COLS, split_frame, to_ref
from src.evaluation import load_annotations
from src.field import Calibration, FieldModel
from src.tracker import TrackerParams

# nome da trajetória -> colunas no log
TRAJ = {"detector": ("raw_x", "raw_y"), "medicao": ("z_x", "z_y"), "estimativa": ("est_x", "est_y")}


# ---------------------------------------------------------------------- trajetórias
def top_candidate(data: dict) -> pd.DataFrame:
    """Detector bruto: o candidato de maior score em cada quadro (quadro atual, px)."""
    rows = []
    for idx, cur, sc in zip(data["frame"], data["cands"], data["scores"]):
        if len(cur):
            k = int(np.argmax(sc))
            rows.append((idx, float(cur[k, 0]), float(cur[k, 1])))
        else:
            rows.append((idx, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["frame", "raw_x", "raw_y"])


def ball_path(ann: pd.DataFrame, data: dict, max_step: int = 10) -> pd.DataFrame:
    """Posição da bola em CADA quadro, interpolada entre marcações visíveis vizinhas.

    As marcações são a cada ~5 quadros. Entre duas marcações consecutivas com a bola visível
    (a no máximo `max_step` quadros uma da outra), a posição é interpolada linearmente no
    quadro de referência (rx, ry) e levada de volta ao quadro atual (cx, cy). Fora desses
    trechos, NaN. Serve para medir quadros não marcados (E3) e escolher trechos.
    """
    frames = np.asarray(data["frame"])
    H = np.asarray(data["H"])
    annr = annotations_ref(ann, dict(zip(data["frame"], data["H"])))
    marks = ann.sort_values("frame")[["frame", "visible"]].to_numpy()
    ok = np.zeros(len(frames), bool)
    for (fa, va), (fb, vb) in zip(marks[:-1], marks[1:]):
        if va == 1 and vb == 1 and fb - fa <= max_step:
            ok[(frames >= fa) & (frames <= fb)] = True
    rx = np.where(ok, np.interp(frames, annr.frame, annr.rx), np.nan)
    ry = np.where(ok, np.interp(frames, annr.frame, annr.ry), np.nan)
    # referência -> quadro atual: H⁻¹ de cada quadro
    Hinv = np.linalg.inv(H)
    p = np.einsum("nij,nj->ni", Hinv, np.column_stack([rx, ry, np.ones(len(rx))]))
    return pd.DataFrame({"frame": frames, "rx": rx, "ry": ry,
                         "cx": p[:, 0] / p[:, 2], "cy": p[:, 1] / p[:, 2]}).set_index("frame")


# ---------------------------------------------------------------------- medidas
def _runs3(frames: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Máscara dos quadros que fecham 3 quadros seguidos com a trajetória presente."""
    consec = np.r_[False, np.diff(frames) == 1]
    return present & np.r_[False, present[:-1]] & np.r_[False, False, present[:-2]] \
        & consec & np.r_[False, consec[:-1]]


def tremor(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Norma da 2ª diferença da posição nos quadros de `mask` (fim de 3 quadros seguidos)."""
    dd = np.full(len(p), np.nan)
    dd[2:] = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1)
    return dd[mask]


def jump_count(frames: np.ndarray, p: np.ndarray, jump_px: float) -> int:
    """Saltos > `jump_px` entre quadros consecutivos com a trajetória presente."""
    ok = np.r_[False, np.diff(frames) == 1]
    d = np.r_[np.nan, np.linalg.norm(np.diff(p, axis=0), axis=1)]
    return int(np.sum(ok & (d > jump_px)))


def _stats(d) -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "mean": round(float(d.mean()), 2), "median": round(float(np.median(d)), 2),
            "p90": round(float(np.quantile(d, .9)), 2)}


def compare(log: pd.DataFrame, raw: pd.DataFrame, ann: pd.DataFrame, annr: pd.DataFrame,
            radius: float = 10.0, jump_px: float = 20.0, f0: int | None = None) -> dict:
    """Medidas das três trajetórias (ver docstring do módulo), a partir do quadro `f0`."""
    lg = log.merge(raw, on="frame", how="left").sort_values("frame").reset_index(drop=True)
    if f0 is not None:
        lg = lg[lg.frame >= f0].reset_index(drop=True)
        ann = ann[ann.frame >= f0]
        annr = annr[annr.frame >= f0]
    frames = lg.frame.to_numpy()
    has_z = lg.z_x.notna().to_numpy()
    zmask = _runs3(frames, has_z)                     # trechos com medição: medição × estimativa
    m = ann.merge(lg, on="frame", how="inner")
    vis, inv = m[m.visible == 1], m[m.visible == 0]
    out = {}
    for name, (xc, yc) in TRAJ.items():
        ref = to_ref(lg, xc, yc)
        present = np.isfinite(ref[:, 0])
        d = np.hypot(vis[xc] - vis.gx, vis[yc] - vis.gy)
        on = d <= radius
        mask = zmask if name != "detector" else _runs3(frames, present)
        bt = pd.DataFrame({"frame": frames, "ref_x": ref[:, 0], "ref_y": ref[:, 1]})
        bt = bt[has_z if name != "detector" else present]
        sig = moving_sigma(bt)
        lag_log = lg[["frame"] + H_COLS].assign(est_x=lg[xc], est_y=lg[yc])
        lag = lag_frames(lag_log, annr)
        out[name] = {
            "coverage_pct": round(100 * float(vis[xc].notna().mean()), 1),
            "on_ball_pct": round(100 * float(on.sum()) / len(vis), 1),
            "error_on_ball_px": _stats(d[on]),
            "output_ball_invisible": None if name == "estimativa" else int(inv[xc].notna().sum()),
            "ball_invisible_frames": int(len(inv)),
            "jumps": jump_count(frames, ref, jump_px),
            "tremor_px": _stats(tremor(ref, mask)),
            "sigma_local_px": round(float(np.hypot(sig["mad_x"], sig["mad_y"]) / np.sqrt(2)), 3)
            if sig.get("n") else None,
            "lag_frames": lag.get("lag_frames"),
        }
    # medição × estimativa nos MESMOS quadros (medição aceita na bola)
    both = vis.dropna(subset=["z_x", "est_x"])
    dz = np.hypot(both.z_x - both.gx, both.z_y - both.gy)
    de = np.hypot(both.est_x - both.gx, both.est_y - both.gy)
    ok = dz <= radius
    out["same_frames"] = {"n": int(ok.sum()), "meas": _stats(dz[ok]), "est": _stats(de[ok])}
    out["visible_frames"] = int(len(vis))
    return out


# ---------------------------------------------------------------------- gráfico
def pick_segment(lg: pd.DataFrame, path: pd.DataFrame, length: int = 90,
                 radius: float = 10.0) -> tuple[int, int]:
    """Trecho com mais quadros em que o detector bruto NÃO está na bola (sem candidato ou
    candidato errado) enquanto a estimativa está: onde o filtro faz diferença."""
    p = path.reindex(lg.frame)
    draw = np.hypot(lg.raw_x.to_numpy() - p.cx.to_numpy(), lg.raw_y.to_numpy() - p.cy.to_numpy())
    dest = np.hypot(lg.est_x.to_numpy() - p.cx.to_numpy(), lg.est_y.to_numpy() - p.cy.to_numpy())
    known = np.isfinite(p.cx.to_numpy())
    good = ~(draw <= radius) & (dest <= radius)      # NaN compara como False: sem candidato conta
    frames = lg.frame.to_numpy()
    best, best_v = int(frames[0]), -1
    for i in range(0, len(frames) - length, 5):
        if known[i:i + length].mean() < 0.7:
            continue
        v = int(good[i:i + length].sum())
        if v > best_v:
            best, best_v = int(frames[i]), v
    return best, best + length


def plot_segment(lg: pd.DataFrame, path: pd.DataFrame, ann: pd.DataFrame, annr: pd.DataFrame,
                 seg: tuple[int, int], out: Path, title: str) -> None:
    """x(t), y(t), plano xy e erro contra a bola: detector bruto × medição × estimativa."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f0, f1 = seg
    s = lg[(lg.frame >= f0) & (lg.frame < f1)]
    raw, z, est = to_ref(s, "raw_x", "raw_y"), to_ref(s, "z_x", "z_y"), to_ref(s, "est_x", "est_y")
    dark = (s.state == "NO_ESCURO").to_numpy()
    a = annr[(annr.frame >= f0) & (annr.frame < f1)]
    p = path.reindex(s.frame)
    t = s.frame.to_numpy()
    est_dark = np.where(dark[:, None], est, np.nan)
    pts = np.vstack([z[np.isfinite(z[:, 0])], a[["rx", "ry"]].to_numpy()])
    lo, hi = (pts.min(axis=0) - 25, pts.max(axis=0) + 25) if len(pts) else (np.zeros(2), np.array([640, 360]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for comp, lab in ((0, "x (px, referência)"), (1, "y (px, referência)")):
        ax = axes[0, comp]
        ax.plot(t, raw[:, comp], ".", ms=5, color="0.55", label="detector bruto (maior score)")
        ax.plot(t, z[:, comp], "o", ms=3.5, color="tab:blue", label="medição aceita")
        ax.plot(t, est[:, comp], "-", lw=1.6, color="tab:orange", label="estimativa (Kalman)")
        ax.plot(t, est_dark[:, comp], "-", lw=2.2, color="tab:red", label="só predição (no escuro)")
        ax.plot(a.frame, a[["rx", "ry"]].to_numpy()[:, comp], "kx", ms=7, label="marcação")
        ax.set_ylim(lo[comp], hi[comp])
        ax.set_xlabel("quadro")
        ax.set_ylabel(lab)
        ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    ax = axes[1, 0]
    ax.plot(raw[:, 0], raw[:, 1], ".", ms=5, color="0.55")
    ax.plot(z[:, 0], z[:, 1], "o", ms=3.5, color="tab:blue")
    ax.plot(est[:, 0], est[:, 1], "-", lw=1.6, color="tab:orange")
    ax.plot(est_dark[:, 0], est_dark[:, 1], "-", lw=2.2, color="tab:red")
    ax.plot(a.rx, a.ry, "kx", ms=7)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(hi[1], lo[1])                       # y da imagem cresce para baixo
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("plano da imagem (pontos do detector fora do recorte não aparecem)", fontsize=9)
    ax.grid(alpha=.3)
    ax = axes[1, 1]
    for (xc, yc), c, lab in ((("raw_x", "raw_y"), "0.55", "detector bruto"),
                             (("est_x", "est_y"), "tab:orange", "estimativa")):
        d = np.hypot(s[xc].to_numpy() - p.cx.to_numpy(), s[yc].to_numpy() - p.cy.to_numpy())
        ax.plot(t, np.clip(d, 0.3, None), ".-" if c != "0.55" else ".", color=c, ms=4, lw=1, label=lab)
    ax.axhline(10, color="k", ls="--", lw=1, label="10 px (na bola)")
    ax.set_yscale("log")
    ax.set_xlabel("quadro")
    ax.set_ylabel("distância à bola (px, log)")
    ax.set_title("distância à posição marcada (interpolada entre marcações)", fontsize=9)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7)
    fig.suptitle(f"{title}: quadros {f0}-{f1}")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- linha de comando
def _rows(r: dict, subset: str) -> list[dict]:
    rows = []
    for name in TRAJ:
        x = r[name]
        rows.append({
            "parte": subset, "trajetoria": name,
            "cobertura%": x["coverage_pct"], "na_bola%": x["on_ball_pct"],
            "erro_med/mediana": f"{x['error_on_ball_px'].get('mean')}/{x['error_on_ball_px'].get('median')}",
            "saidas_bola_invis": "-" if x["output_ball_invisible"] is None
            else f"{x['output_ball_invisible']}/{x['ball_invisible_frames']}",
            "saltos>20": x["jumps"],
            "tremor_med/p90": f"{x['tremor_px'].get('median')}/{x['tremor_px'].get('p90')}",
            "sigma_local": x["sigma_local_px"], "atraso_q": x["lag_frames"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--metric", action="store_true", help="filtro F2 (metros)")
    ap.add_argument("--segment", type=int, nargs=2, default=None, help="quadros do gráfico")
    ap.add_argument("--sigma-m", type=float, default=TrackerParams.sigma_m)
    args = ap.parse_args()

    stem = Path(args.video).stem
    scorer = args.scorer or f"data/models/ball_scorer_{stem}.json"
    exp_dir = Path("outputs/experiments") / stem
    data = collect(args.video, scorer, cache=exp_dir / "replay_cache.pkl")
    pw = data["size"][0]
    ann = load_annotations(Path("data/annotations") / f"{stem}.csv", scale=pw / data["orig_width"])
    annr = annotations_ref(ann, dict(zip(data["frame"], data["H"])))
    split = split_frame(ann)
    field_path = Path("data/calib") / f"{stem}_field.json"
    field = FieldModel(Calibration.load(field_path), pw) if field_path.exists() else None
    if args.metric and field is None:
        raise SystemExit("--metric precisa da calibração do campo")
    mode = "F2" if args.metric else "F1"

    tp = TrackerParams.metric(fps=data["fps"]) if args.metric else TrackerParams(fps=data["fps"], sigma_m=args.sigma_m)
    log = replay(data, tp, field, args.metric, args.sigma_m)
    raw = top_candidate(data)
    res = {"teste": compare(log, raw, ann, annr, f0=split), "todo": compare(log, raw, ann, annr)}
    res["split_frame"] = split
    res["params"] = {"filter": mode, "sigma_a": tp.sigma_a, "maneuver_d2": tp.maneuver_d2,
                     "maneuver_factor": tp.maneuver_factor, "max_dark": tp.max_dark}

    lg = log.merge(raw, on="frame", how="left")
    path = ball_path(ann, data)
    seg = tuple(args.segment) if args.segment else pick_segment(lg, path)
    png = exp_dir / f"e1_{mode}_trecho_{seg[0]}.png"
    plot_segment(lg, path, ann, annr, seg, png, f"E1 {stem} {mode}")

    df = pd.DataFrame(_rows(res["teste"], "teste") + _rows(res["todo"], "todo"))
    out_csv = exp_dir / f"e1_{mode}.csv"
    df.to_csv(out_csv, index=False)
    (exp_dir / f"e1_{mode}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                                             encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.to_string(index=False))
    for part in ("teste", "todo"):
        sf = res[part]["same_frames"]
        print(f"  {part}, mesmos quadros (n={sf['n']}): erro mediano medicao {sf['meas'].get('median')} px, "
              f"estimativa {sf['est'].get('median')} px")
    print(f"tabela em {out_csv}; grafico em {png}")


if __name__ == "__main__":
    main()
