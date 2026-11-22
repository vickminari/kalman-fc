"""E2 — três valores de Σd (σa): teimoso, intermediário e nervoso. Também mede o ruído de
processo adaptativo (detecção de manobra) contra o σa fixo.

Para cada configuração, roda a trilha sobre os candidatos em cache
(`experiments/replay_tracker.py`) e mede, contra as marcações:

- **K da posição em regime** (o "botão" elefante × peixinho-dourado);
- **"na bola"** (teste e vídeo todo) e saltos/associação da régua (report 011);
- **erro** da medição crua e da estimativa, nos mesmos quadros (bola visível, medição
  aceita na bola): é o E1 dentro de cada σa;
- **tremor** (2ª diferença) da medição e da estimativa;
- **atraso:** o deslocamento temporal s (quadros) que minimiza o erro mediano entre a
  estimativa no quadro f + s e a marcação no quadro f. Filtro teimoso → a estimativa chega
  depois da bola (s > 0). Medido só com a bola em movimento (a parada não tem atraso) e
  nos quadros em que a trilha está na bola (senão o erro mede troca de alvo, não atraso).

Tudo em px a 640, no quadro de referência (estabilizado), também no F2.

Gráficos (`outputs/experiments/<vídeo>/e2_<F1|F2>_trecho.png`): trajetória crua × filtrada
num trecho com chute, para as três configurações de σa.

Uso (a partir da raiz do repositório):

    python -m experiments.e2_sigma_a videos/campo_lateral.MOV
    python -m experiments.e2_sigma_a videos/campo_lateral.MOV --metric --sigma-a 10 40 150
    python -m experiments.e2_sigma_a videos/campo_centro.MOV --segment 2400 2460
    python -m experiments.e2_sigma_a videos/campo_lateral.MOV --adaptive 1000:6:20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.replay_tracker import collect, replay
from experiments.track_diagnostics import diagnose, split_frame, to_ref, typical_Sm
from src.evaluation import load_annotations
from src.field import Calibration, FieldModel
from src.stabilize import Stabilizer
from src.tracker import BallTracker, TrackerParams


# ---------------------------------------------------------------------- marcações na referência
def annotations_ref(ann: pd.DataFrame, H_by_frame: dict) -> pd.DataFrame:
    """Marcações visíveis levadas ao quadro de referência, com a velocidade (px/quadro)
    estimada pelas marcações vizinhas."""
    v = ann[(ann.visible == 1) & ann.frame.isin(list(H_by_frame))].sort_values("frame").copy()
    pts = np.array([Stabilizer.apply(H_by_frame[int(f)], np.array([[x, y]]))[0]
                    for f, x, y in zip(v.frame, v.gx, v.gy)]).reshape(-1, 2)
    v["rx"], v["ry"] = pts[:, 0], pts[:, 1]
    f = v.frame.to_numpy()
    vel = np.full((len(v), 2), np.nan)
    for i in range(1, len(v) - 1):
        if f[i + 1] - f[i - 1] <= 12:
            vel[i] = (pts[i + 1] - pts[i - 1]) / (f[i + 1] - f[i - 1])
    v["vx"], v["vy"] = vel[:, 0], vel[:, 1]
    v["speed"] = np.hypot(vel[:, 0], vel[:, 1])
    return v


def lag_frames(log: pd.DataFrame, annr: pd.DataFrame, max_shift: int = 6, min_speed: float = 2.0,
               on_ball: float = 15.0) -> dict:
    """Atraso da estimativa: s que minimiza a mediana de |est(f+s) − marcação(f)|.

    Usa as marcações com a bola andando mais que `min_speed` px/quadro e com a estimativa
    a ≤ `on_ball` px da marcação em s = 0 (trilha na bola). Todos os s usam os mesmos quadros.
    O mínimo é refinado por uma parábola nos 3 pontos em volta (atraso fracionário).
    """
    est = pd.DataFrame(to_ref(log, "est_x", "est_y"), columns=["ex", "ey"], index=log.frame)
    a = annr[annr.speed > min_speed]
    shifts = np.arange(-max_shift, max_shift + 1)
    rows = []
    for f, rx, ry in zip(a.frame, a.rx, a.ry):
        e = [est.loc[f + s].to_numpy() if (f + s) in est.index else np.array([np.nan, np.nan]) for s in shifts]
        d = np.hypot(*(np.array(e) - [rx, ry]).T)
        if np.isfinite(d).all() and d[max_shift] <= on_ball:
            rows.append(d)
    if len(rows) < 5:
        return {"n": len(rows)}
    D = np.array(rows)
    med = np.median(D, axis=0)
    k = int(np.argmin(med))
    frac = 0.0
    if 0 < k < len(shifts) - 1:
        y0, y1, y2 = med[k - 1], med[k], med[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den > 0 else 0.0
    return {"n": int(len(D)), "lag_frames": round(float(shifts[k] + frac), 2),
            "err_at_0_px": round(float(med[max_shift]), 2), "err_at_best_px": round(float(med[k]), 2),
            "median_err_by_shift": {int(s): round(float(m), 2) for s, m in zip(shifts, med)}}


def raw_vs_filtered(log: pd.DataFrame, ann: pd.DataFrame, radius: float = 10.0) -> dict:
    """E1 dentro de cada σa: erro da medição crua × da estimativa, nos MESMOS quadros
    (bola visível, medição aceita a ≤ `radius` px da marcação; quadro atual)."""
    m = ann[ann.visible == 1].merge(log, on="frame").dropna(subset=["z_x", "est_x"])
    dz = np.hypot(m.z_x - m.gx, m.z_y - m.gy)
    de = np.hypot(m.est_x - m.gx, m.est_y - m.gy)
    ok = dz <= radius
    return {"n": int(ok.sum()), "meas_median_px": round(float(np.median(dz[ok])), 2),
            "est_median_px": round(float(np.median(de[ok])), 2),
            "meas_mean_px": round(float(dz[ok].mean()), 2), "est_mean_px": round(float(de[ok].mean()), 2)}


# ---------------------------------------------------------------------- resposta a um chute (sintético)
def step_response(params: TrackerParams, v0: float = 100.0, v1: float = 600.0, kick: int = 30,
                  n: int = 70, noise: float = 0.0, seed: int = 0) -> dict:
    """Bola em x a `v0` px/s, chutada para `v1` px/s no quadro `kick`. Mede o atraso do filtro.

    Sem ruído (`noise` = 0), o que sobra é só a dinâmica do filtro:
    - quadros até a velocidade estimada chegar a 90% do salto (v0 -> v1);
    - erro máximo de posição depois do chute;
    - quadros no escuro (o chute tirou a bola da região de busca).
    Com ruído (σ = `noise` px por eixo), mede o tremor da estimativa antes do chute.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / params.fps
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + (v0 if t <= kick else v1) * dt
    truth = np.column_stack([100 + x, np.full(n, 200.0)])
    meas = truth + rng.normal(0, noise, truth.shape) if noise > 0 else truth
    tr = BallTracker(params)
    est, vel, dark = [], [], 0
    for t in range(n):
        s = tr.step(meas[t:t + 1])
        est.append(s.x[:2] if s.x is not None else np.array([np.nan, np.nan]))
        vel.append(float(s.x[2]) if s.x is not None else np.nan)
        dark += int(s.z is None)
    est, vel = np.array(est), np.array(vel)
    after = np.arange(kick + 1, n)
    reach = np.flatnonzero(vel[after] >= v0 + 0.9 * (v1 - v0))
    err = np.linalg.norm(est - truth, axis=1)
    dd = est[2:kick] - 2 * est[1:kick - 1] + est[:kick - 2]
    return {"settle_frames": int(reach[0] + 1) if len(reach) else None,
            "max_err_after_kick_px": round(float(err[after].max()), 2),
            "dark_frames": dark, "velocity": vel.tolist(),
            "tremor_before_px": round(float(np.median(np.linalg.norm(dd[5:], axis=1))), 3) if noise > 0 else None}


# cores fixas por posição na lista de configurações (teimoso, intermediário, nervoso, adaptativo)
COLORS = ["tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown", "tab:pink"]


def plot_step(responses: dict[str, dict[int, dict]], out: Path, v0: float = 100.0,
              kick: int = 30) -> None:
    """Velocidade estimada depois de chutes de tamanhos diferentes (um painel por chute)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kicks = sorted(next(iter(responses.values())))
    fig, axes = plt.subplots(1, len(kicks), figsize=(6.5 * len(kicks), 4.5), squeeze=False)
    for ax, dv in zip(axes[0], kicks):
        n = len(next(iter(responses.values()))[dv]["velocity"])
        t = np.arange(n)
        ax.plot(t, np.where(t <= kick, v0, v0 + dv), "k--", lw=1, label="velocidade real")
        for (name, r), c in zip(responses.items(), COLORS):
            st = r[dv]
            lab = f"{st['settle_frames']} q" if st["dark_frames"] < 10 else "perde a bola"
            ax.plot(t, st["velocity"], lw=1.6, color=c, label=f"{name}: {lab}")
        ax.set_xlim(kick - 5, kick + 25)
        ax.set_ylim(v0 - 150, v0 + dv + 250)
        ax.set_xlabel(f"quadro (chute no quadro {kick})")
        ax.set_ylabel("velocidade estimada vx (px/s)")
        ax.set_title(f"chute de +{dv} px/s ({dv / 30:.1f} px/quadro)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle("Resposta a um chute (sintético, sem ruído): quadros até 90% da nova velocidade")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- trecho para o gráfico
def pick_segment(log: pd.DataFrame, annr: pd.DataFrame, length: int = 45) -> tuple[int, int]:
    """Trecho de `length` quadros com a maior MUDANÇA de velocidade da bola (um chute, pelas
    marcações) em que a trilha tem medição em pelo menos 80% dos quadros."""
    has_z = log.set_index("frame").z_x.notna()
    a = annr.dropna(subset=["vx"]).sort_values("frame")
    acc = np.r_[np.nan, np.hypot(np.diff(a.vx), np.diff(a.vy))]
    acc[np.r_[False, np.diff(a.frame) > 10]] = np.nan          # só marcações vizinhas
    best, best_v = None, -1.0
    for fk, ak in zip(a.frame, acc):
        if not np.isfinite(ak) or ak <= best_v:
            continue
        f0 = int(fk) - length // 2
        sel = has_z.loc[f0:f0 + length - 1]
        if len(sel) == length and sel.mean() >= 0.8:
            best, best_v = f0, ak
    if best is None:
        best = int(log.frame.min())
    return best, best + length


def plot_segment(logs: dict[str, pd.DataFrame], annr: pd.DataFrame, seg: tuple[int, int],
                 out: Path, title: str) -> None:
    """Trajetória crua (medições) × filtrada (uma linha por configuração) num trecho."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f0, f1 = seg
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    first = next(iter(logs.values()))
    s0 = first[(first.frame >= f0) & (first.frame < f1)]
    z = to_ref(s0, "z_x", "z_y")
    zc = z[np.isfinite(z[:, 0])]
    a = annr[(annr.frame >= f0) & (annr.frame < f1)]
    A = a[["rx", "ry"]].to_numpy()
    if len(zc):                                    # marcações longe das medições achatam a escala
        near = np.array([np.min(np.hypot(*(zc - p).T)) < 30 for p in A], bool)
        a, A = a[near], A[near]
        lo, hi = zc.min(axis=0) - 15, zc.max(axis=0) + 15
    else:
        lo, hi = np.array([0, 0]), np.array([640, 360])
    for comp, lab in ((0, "x (px, referência)"), (1, "y (px, referência)")):
        ax = axes[comp]
        ax.plot(s0.frame, z[:, comp], "o", ms=3, color="tab:blue", label="medição (crua)")
        ax.plot(a.frame, A[:, comp], "kx", ms=7, label="marcação")
        for (name, log), c in zip(logs.items(), COLORS):
            s = log[(log.frame >= f0) & (log.frame < f1)]
            e = to_ref(s, "est_x", "est_y")
            ax.plot(s.frame, e[:, comp], "-", color=c, lw=1.4, label=name)
        ax.set_ylim(lo[comp], hi[comp])
        ax.set_xlabel("quadro")
        ax.set_ylabel(lab)
        ax.grid(alpha=.3)
    ax = axes[2]
    ax.plot(z[:, 0], z[:, 1], "o", ms=3, color="tab:blue", label="medição (crua)")
    ax.plot(A[:, 0], A[:, 1], "kx", ms=7, label="marcação")
    for (name, log), c in zip(logs.items(), COLORS):
        s = log[(log.frame >= f0) & (log.frame < f1)]
        e = to_ref(s, "est_x", "est_y")
        ax.plot(e[:, 0], e[:, 1], "-", color=c, lw=1.4, label=name)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(hi[1], lo[1])                      # y da imagem cresce para baixo
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.suptitle(f"{title}: quadros {f0}-{f1}")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- linha de comando
def parse_adaptive(spec: str) -> dict:
    """"sigma_a:d2:fator[:quadros[:escuro]]" -> parâmetros do ruído de processo adaptativo
    (escuro = 1 também aumenta Σd nas predições sem medição)."""
    parts = [float(p) for p in spec.split(":")]
    out = {"sigma_a": parts[0], "maneuver_d2": parts[1], "maneuver_factor": parts[2]}
    if len(parts) > 3:
        out["maneuver_frames"] = int(parts[3])
    if len(parts) > 4:
        out["maneuver_in_dark"] = bool(parts[4])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--metric", action="store_true")
    ap.add_argument("--sigma-a", type=float, nargs="+", default=None,
                    help="padrão: 300 1000 4000 px/s² (F1) ou 10 40 150 m/s² (F2)")
    ap.add_argument("--adaptive", nargs="*", default=[],
                    help="configurações adaptativas 'sigma_a:d2:fator[:quadros]' (ver src/tracker.py)")
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
    unit = "m/s2" if args.metric else "px/s2"
    sigmas = args.sigma_a or ([10.0, 40.0, 150.0] if args.metric else [300.0, 1000.0, 4000.0])
    labels = ["teimoso", "intermediario", "nervoso"] if len(sigmas) == 3 else [f"sa{s:g}" for s in sigmas]

    # σa fixo: a detecção de manobra fica desligada (maneuver_d2 = 0)
    configs = [(f"{lab} (sa={s:g})", {"sigma_a": s, "maneuver_d2": 0.0}) for lab, s in zip(labels, sigmas)]
    for spec in args.adaptive:
        kw = parse_adaptive(spec)
        extra = f", {kw.get('maneuver_frames', 1)}q" + (", escuro" if kw.get("maneuver_in_dark") else "")
        configs.append((f"adaptativo (sa={kw['sigma_a']:g}, d2>{kw['maneuver_d2']:g}, x{kw['maneuver_factor']:g}{extra})", kw))

    rows, logs, full, steps = [], {}, {}, {}
    for name, kw in configs:
        tp = TrackerParams.metric(fps=data["fps"], **kw) if args.metric else \
            TrackerParams(fps=data["fps"], sigma_m=args.sigma_m, **kw)
        if not args.metric:
            # resposta a um chute, em px (sintético): o atraso "de livro" de cada σa
            steps[name] = {dv: step_response(tp, v1=100 + dv) for dv in (200, 500)}
            st = steps[name][500]
            st_noise = step_response(tp, noise=0.3)
        log = replay(data, tp, field, args.metric, args.sigma_m)
        Sm = typical_Sm(log, args.sigma_m, field if args.metric else None)
        d = diagnose(log, ann, data["fps"], tp.sigma_a, Sm, split=split)
        lag = lag_frames(log, annr)
        lag_t = lag_frames(log[log.frame >= split], annr[annr.frame >= split])
        rv = raw_vs_filtered(log, ann)
        logs[name] = log
        full[name] = {"params": kw, "diagnostics": d, "lag": lag, "lag_test": lag_t, "raw_vs_filtered": rv}
        g = d["kalman_gain"]
        rows.append({
            "config": name,
            "K_pos": f"{g['K_pos_min']:.2f}" if g["K_pos_min"] == g["K_pos_max"] else f"{g['K_pos_min']:.2f}-{g['K_pos_max']:.2f}",
            "bola_teste%": d["reference"]["on_ball_test_pct"], "bola_todo%": d["reference"]["on_ball_all_pct"],
            "erro_med/est": f"{rv['meas_median_px']}/{rv['est_median_px']}",
            "tremor_med/est": f"{d['tremor_px']['measurement']['median']}/{d['tremor_px']['estimate']['median']}",
            "atraso_q": lag.get("lag_frames"), "erro_s0/smin": f"{lag.get('err_at_0_px')}/{lag.get('err_at_best_px')}",
            "saltos": d["jumps"]["total"], "s_escuro": d["jumps"]["no_escuro"],
            "escuro_err_teste": d["reference"]["dark_pred_err_test_median_px"],
            "aceit_bola": f"{d['association']['accepted_on_ball']}/{d['association']['accepted_visible']}",
            "trilhas": d["association"]["tracks_started"],
            "escuro_rec_10_30": d["dark_episodes"]["recovered_10_30"],
        })
        if not args.metric:
            s2 = steps[name][200]
            rows[-1].update({"chute200_q90%/escuro": f"{s2['settle_frames']}/{s2['dark_frames']}",
                             "chute500_q90%/escuro": f"{st['settle_frames']}/{st['dark_frames']}",
                             "sint_tremor_est": st_noise["tremor_before_px"]})
            full[name]["step_response"] = {dv: {k: v for k, v in r.items() if k != "velocity"}
                                           for dv, r in steps[name].items()}
            full[name]["step_response_tremor_0.3px"] = st_noise["tremor_before_px"]
        print(f"  {name}: na bola (teste) {d['reference']['on_ball_test_pct']}%", flush=True)

    df = pd.DataFrame(rows).set_index("config")
    out_csv = exp_dir / f"e2_{mode}.csv"
    df.to_csv(out_csv)
    (exp_dir / f"e2_{mode}.json").write_text(json.dumps(full, indent=2, ensure_ascii=False, default=float),
                                             encoding="utf-8")
    # trecho do gráfico: escolhido sobre a trilha mais nervosa (a que mais tem medição)
    nervous = configs[len(sigmas) - 1][0]
    seg = tuple(args.segment) if args.segment else pick_segment(logs[nervous], annr)
    png = exp_dir / f"e2_{mode}_trecho_{seg[0]}.png"
    plot_segment(logs, annr, seg, png, f"E2 {stem} {mode} (sigma_a em {unit})")
    if steps:
        plot_step(steps, exp_dir / "e2_F1_chute_sintetico.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.T.to_string())
    print(f"tabela em {out_csv}; grafico em {png}")


if __name__ == "__main__":
    main()
