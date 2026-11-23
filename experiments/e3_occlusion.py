"""E3 — oclusão de 10 a 30 quadros (experimento obrigatório).

Duas partes, sobre os candidatos em cache (`experiments/replay_tracker.py`), com os
parâmetros padrão da trilha (F1 em px e F2 em metros):

**1. Oclusão simulada (controlada).** Em trechos em que a bola está visível, detectada e
rastreada, a bola é "tapada" por N quadros (N = 10, 20, 30): os candidatos a até
`--remove-radius` px da posição marcada (interpolada entre marcações) são apagados. Dois modos:

- `bola`: só a bola some; os outros candidatos (pés, calções) continuam. É a oclusão
  realista (a bola passa atrás de um jogador) e testa se o gating evita aceitar um falso;
- `tudo`: nenhum candidato no quadro (apagão do detector).

Por oclusão, mede:

- **quadros recuperados:** quadros do buraco em que a predição (mesma trilha) fica a
  ≤ 10 px e a ≤ 20 px da bola;
- **erro da predição k quadros depois do início** (curva por k; é também a "predição k
  quadros à frente" do enunciado);
- **trilha mantida:** sem perder a trilha e sem trocar de trilha durante o buraco
  (com N > `max_dark` = 20, a trilha é perdida de propósito: é o limite do enunciado);
- **associações erradas:** medições aceitas dentro do buraco (todas são falsas: a bola foi apagada);
- **reaceitação:** quadros até a primeira medição aceita na bola depois do buraco, e se é a
  MESMA trilha (o filtro retomou a correção) ou uma trilha nova (reinício).

Também calcula, pela covariância, como o **raio da região de busca cresce** no escuro
(a partir do regime) e onde o teto de `max_gate_px` corta.

**2. Oclusões naturais.** Os episódios no escuro de 10–30 quadros da trilha normal, com o
motivo pelas marcações (bola marcada como invisível = oclusão ou saída; bola visível =
falha do detector), o erro da predição nos quadros marcados e se a bola foi reaceita.
A lista (com o tempo em mm:ss) serve para escolher o trecho de oclusão do vídeo de entrega.

Saídas em `outputs/experiments/<vídeo>/`: `e3_simulada.csv` (uma linha por oclusão),
`e3_resumo.csv`, `e3_naturais.csv`, `e3.json`, `e3_erro_k.png` (erro × k e raio da região
de busca) e `e3_exemplo_<quadro>.png` (uma oclusão de 20 quadros, F1, com as elipses).

Uso (a partir da raiz do repositório):

    python -m experiments.e3_occlusion videos/campo_lateral.MOV
    python -m experiments.e3_occlusion videos/campo_centro.MOV --n 10 20 30 --max-dark 20 30
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e1_raw_vs_filtered import ball_path
from experiments.replay_tracker import collect, replay
from experiments.track_diagnostics import to_ref
from src.evaluation import load_annotations
from src.field import Calibration, FieldModel
from src.gating import chi2_threshold, ellipse
from src.kalman import constant_velocity, steady_state_gain
from src.tracker import TrackerParams


# ---------------------------------------------------------------------- escolha das oclusões
def choose_gaps(logs: list[pd.DataFrame], path: pd.DataFrame, n: int, pre: int = 3, post: int = 10,
                spacing: int = 60, radius: float = 10.0, min_detected: float = 0.5) -> list[int]:
    """Inícios de oclusão de `n` quadros onde a simulação é justa, em TODAS as trilhas de `logs`:

    - posição da bola conhecida (marcações visíveis vizinhas) de s − `pre` até s + n + `post`;
    - no quadro s − 1, a trilha está RASTREANDO com a medição na bola;
    - na trilha normal, a bola foi medida em ≥ `min_detected` dos quadros do buraco (a
      oclusão tira de fato uma bola que estava sendo vista);
    - na trilha normal, a bola é medida de novo até `post` quadros depois do buraco (se não
      fosse, a falha na reaceitação seria do detector, não da oclusão);
    - `spacing` quadros entre o fim de uma oclusão e o início da próxima (a trilha se refaz).
    """
    frames = path.index.to_numpy()
    known = np.isfinite(path.cx.to_numpy())
    on = []
    for log in logs:
        lg = log.set_index("frame").reindex(frames)
        dz = np.hypot(lg.z_x.to_numpy() - path.cx.to_numpy(), lg.z_y.to_numpy() - path.cy.to_numpy())
        on.append((dz <= radius, (lg.state == "RASTREANDO").to_numpy()))
    starts, free_from = [], -np.inf
    for s in range(pre + 1, len(frames) - n - post):
        if s < free_from or not known[s - pre:s + n + post].all():
            continue
        ok = all(trk[s - 1] and z[s - 1] and z[s:s + n].mean() >= min_detected and z[s + n:s + n + post].any()
                 for z, trk in on)
        if ok:
            starts.append(int(frames[s]))
            free_from = s + n + post + spacing
    return starts


def occlude(data: dict, starts: list[int], n: int, path: pd.DataFrame, mode: str = "bola",
            radius: float = 20.0) -> dict:
    """Cópia do cache com a bola apagada por `n` quadros a partir de cada início.

    `bola`: tira os candidatos a ≤ `radius` px da posição da bola (quadro atual);
    `tudo`: tira todos os candidatos.
    """
    assert list(data["frame"]) == list(range(len(data["frame"]))), "o cache tem de começar no quadro 0"
    cands, scores = list(data["cands"]), list(data["scores"])
    for s in starts:
        for f in range(s, s + n):
            c, sc = cands[f], scores[f]
            if len(c) == 0:
                continue
            if mode == "tudo":
                keep = np.zeros(len(c), bool)
            else:
                keep = np.hypot(c[:, 0] - path.cx.loc[f], c[:, 1] - path.cy.loc[f]) > radius
            cands[f], scores[f] = c[keep], sc[keep]
    return {**data, "cands": cands, "scores": scores}


# ---------------------------------------------------------------------- medidas
def gap_metrics(log: pd.DataFrame, steps: dict, starts: list[int], n: int, path: pd.DataFrame,
                threshold: float, radius: float = 10.0, post: int = 30) -> list[dict]:
    """Uma linha por oclusão (ver docstring do módulo). Erros em px, no quadro de referência."""
    lg = log.sort_values("frame").reset_index(drop=True)
    assert (lg.frame.to_numpy() == np.arange(len(lg))).all()
    est = to_ref(lg, "est_x", "est_y")
    err = np.hypot(est[:, 0] - path.rx.to_numpy(), est[:, 1] - path.ry.to_numpy())
    dz = np.hypot(lg.z_x.to_numpy() - path.cx.to_numpy(), lg.z_y.to_numpy() - path.cy.to_numpy())
    new = lg.new_track.astype(bool).to_numpy()
    state = lg.state.to_numpy()
    has_z = lg.z_x.notna().to_numpy()
    rows = []
    for s in starts:
        g = np.arange(s, s + n)
        # a predição só conta enquanto é a MESMA trilha que vinha na bola
        same = np.cumsum(new[g]) == 0
        alive = same & (state[g] != "BUSCANDO")
        e = np.where(alive, err[g], np.nan)
        lost = not alive.all()
        after = np.arange(s + n, min(s + n + post, len(lg)))
        hit = after[dz[after] <= radius]
        reacq = int(hit[0] - (s + n)) if len(hit) else None
        same_track = reacq is not None and not new[s:hit[0] + 1].any()
        st = steps.get(s + n - 1)
        gate = None
        if st is not None and st.S is not None:
            gate = float(np.sqrt(np.linalg.eigvalsh(st.S).max() * threshold))
        rows.append({
            "start": s, "n": n,
            # válida se, nesta rodada, a trilha chegou à oclusão rastreando a bola (uma oclusão
            # anterior pode ter deixado a trilha num falso)
            "valid": bool(state[s - 1] == "RASTREANDO" and dz[s - 1] <= radius),
            "kept": not lost, "lost_at_k": int(np.argmin(alive)) + 1 if lost else None,
            "covered_10": int(np.sum(e <= 10)), "covered_20": int(np.sum(e <= 20)),
            "wrong_accepts": int(has_z[g].sum()),
            "err_end_px": round(float(e[-1]), 2) if np.isfinite(e[-1]) else None,
            "err_median_px": round(float(np.nanmedian(e)), 2) if np.isfinite(e).any() else None,
            "reacq_frames": reacq, "reacq_same_track": bool(same_track),
            "gate_end": round(gate, 3) if gate is not None else None,
            "err_k": [round(float(x), 2) if np.isfinite(x) else None for x in e],
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    """Resumo de um conjunto de oclusões (mesmos N, modo e filtro); só as válidas."""
    df = pd.DataFrame(rows)
    invalid = int((~df.valid).sum())
    df = df[df.valid]
    if df.empty:
        return {"oclusoes": 0, "invalidas": invalid}
    n = int(df.n.iloc[0])
    ek = np.array([[np.nan if x is None else x for x in r] for r in df.err_k], float)

    def at(k):
        # mediana só com pelo menos 3 oclusões ainda na mesma trilha naquele k
        col = ek[:, k - 1] if k <= ek.shape[1] else np.full(len(ek), np.nan)
        return round(float(np.nanmedian(col)), 1) if np.isfinite(col).sum() >= 3 else None

    end = df.err_end_px.dropna()
    re = df.reacq_frames.dropna()
    return {
        "oclusoes": int(len(df)), "invalidas": invalid,
        "mantida%": round(100 * float(df.kept.mean()), 1),
        "recup_10px%": round(100 * float(df.covered_10.sum()) / (n * len(df)), 1),
        "recup_20px%": round(100 * float(df.covered_20.sum()) / (n * len(df)), 1),
        "erro_k5": at(5), "erro_k10": at(10), "erro_k20": at(20), "erro_k30": at(30),
        "erro_fim_mediano": round(float(end.median()), 1) if len(end) >= 3 else None,
        "aceit_erradas": int(df.wrong_accepts.sum()),
        "oclusoes_c_aceit_errada": int((df.wrong_accepts > 0).sum()),
        "reaceita%": round(100 * float(df.reacq_frames.notna().mean()), 1),
        "reaceita_mesma_trilha%": round(100 * float(df.reacq_same_track.mean()), 1),
        "quadros_ate_reaceitar": round(float(re.median()), 1) if len(re) else None,
    }


def natural_episodes(log: pd.DataFrame, ann: pd.DataFrame, path: pd.DataFrame, fps: float,
                     min_len: int = 10, max_len: int = 30, radius: float = 10.0) -> pd.DataFrame:
    """Episódios NO_ESCURO de `min_len`–`max_len` quadros na trilha normal, com o motivo.

    `na_bola_antes`: a última medição antes do escuro estava na bola (senão o episódio é de
    uma trilha que já seguia um falso, e não uma oclusão da bola).
    """
    lg = log.sort_values("frame").reset_index(drop=True)
    st = (lg.state == "NO_ESCURO").to_numpy()
    has_z = lg.z_x.notna().to_numpy()
    new = lg.new_track.astype(bool).to_numpy()
    a = ann.set_index("frame")
    rows, i = [], 0
    while i < len(st):
        if not st[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(st) and st[j + 1]:
            j += 1
        L = j - i + 1
        if min_len <= L <= max_len:
            f0, f1 = int(lg.frame[i]), int(lg.frame[j])
            marks = a.loc[(a.index >= f0) & (a.index <= f1)]
            vis, inv = int((marks.visible == 1).sum()), int((marks.visible == 0).sum())
            if inv > vis:
                why = "bola invisivel (oclusao ou fora)"
            elif vis > 0:
                why = "bola visivel (falha do detector)"
            else:
                why = "sem marcacao"
            mv = marks[marks.visible == 1].join(lg.set_index("frame")[["est_x", "est_y"]])
            e = np.hypot(mv.est_x - mv.gx, mv.est_y - mv.gy)
            k = j + 1
            before = None
            fb = int(lg.frame[i - 1]) if i > 0 else None
            if fb is not None and has_z[i - 1] and np.isfinite(path.cx.loc[fb]):
                before = bool(np.hypot(lg.z_x[i - 1] - path.cx.loc[fb], lg.z_y[i - 1] - path.cy.loc[fb]) <= radius)
            recovered = k < len(st) and has_z[k] and not new[k]
            on_ball = None
            if recovered and np.isfinite(path.cx.loc[int(lg.frame[k])]):
                on_ball = bool(np.hypot(lg.z_x[k] - path.cx.loc[int(lg.frame[k])],
                                        lg.z_y[k] - path.cy.loc[int(lg.frame[k])]) <= radius)
            rows.append({"inicio": f0, "tempo": f"{int(f0 / fps // 60)}:{f0 / fps % 60:04.1f}",
                         "quadros": L, "motivo": why, "na_bola_antes": before, "marcas_visiveis": vis, "marcas_invisiveis": inv,
                         "erro_pred_mediano_px": round(float(e.median()), 1) if len(e) else None,
                         "recuperada": bool(recovered), "reaceita_na_bola": on_ball})
        i = j + 1
    return pd.DataFrame(rows)


def gate_growth(tp: TrackerParams, n: int, Sm: np.ndarray | None = None) -> np.ndarray:
    """Raio maior da região de busca (unidades do filtro) após k = 1..n predições sem medição,
    partindo da covariância em regime (medição em todo quadro). Sem o teto de `max_gate_px`."""
    m = constant_velocity(1.0 / tp.fps, tp.sigma_a, tp.sigma_m)
    Sm = m.Sm if Sm is None else Sm
    K, Pp = steady_state_gain(m.D, m.M, m.Sd, Sm)
    P = (np.eye(4) - K @ m.M) @ Pp                   # P⁺ em regime
    thr = chi2_threshold(tp.gate_prob)
    r = []
    for _ in range(n):
        P = m.D @ P @ m.D.T + m.Sd
        S = m.M @ P @ m.M.T + Sm
        r.append(np.sqrt(np.linalg.eigvalsh(S).max() * thr))
    return np.array(r)


# ---------------------------------------------------------------------- gráficos
def plot_error_k(curves: dict[str, list[dict]], gate: np.ndarray, gate_cap: float, max_dark: int,
                 out: Path, title: str) -> None:
    """Erro da predição × quadros de oclusão (mediana e quartis) e o raio da região de busca."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    # o primeiro max_dark de cada filtro: linha cheia com quartis; os outros: tracejada, só depois
    # do primeiro max_dark (antes disso as curvas são iguais)
    first_md = {}
    for (name, rows), c in zip(curves.items(), ["tab:orange", "tab:purple", "tab:green", "tab:brown"]):
        rows = [r for r in rows if r.get("valid", True)]
        filt = name.split(",")[0]
        main_curve = filt not in first_md
        first_md.setdefault(filt, True)
        ek = np.array([[np.nan if x is None else x for x in r["err_k"]] for r in rows], float)
        k = np.arange(1, ek.shape[1] + 1)
        ok = np.isfinite(ek).sum(axis=0) >= 3        # k com pelo menos 3 oclusões na mesma trilha
        if not main_curve:
            ok &= k >= max_dark
        med, q1, q3 = (np.full(len(k), np.nan) for _ in range(3))
        med[ok] = np.nanmedian(ek[:, ok], axis=0)
        q1[ok], q3[ok] = np.nanpercentile(ek[:, ok], [25, 75], axis=0)
        ax.plot(k, med, "-" if main_curve else "--", color=c, lw=2, label=f"erro da predição, {name} (mediana)")
        if main_curve:
            ax.fill_between(k, q1, q3, color=c, alpha=.2)
    kk = np.arange(1, len(gate) + 1)
    ax.plot(kk, gate, "--", color="0.5", lw=1.2, label="raio da região de busca F1 (pela covariância)")
    ax.plot(kk, np.minimum(gate, gate_cap), "-", color="0.3", lw=1.2, label=f"raio usado (teto {gate_cap:g} px)")
    ax.axhline(10, color="k", ls=":", lw=1, label="10 px")
    ax.axvline(max_dark, color="tab:red", ls="--", lw=1, label=f"max_dark = {max_dark} (trilha perdida)")
    ax.set_xlabel("quadros desde o início da oclusão (k)")
    ax.set_ylabel("px (640 de largura, referência)")
    ax.set_ylim(0, max(80, gate_cap * 1.3))
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_example(log: pd.DataFrame, steps: dict, path: pd.DataFrame, annr: pd.DataFrame, s: int, n: int,
                 threshold: float, out: Path, title: str, pre: int = 15, post: int = 15) -> None:
    """Uma oclusão (F1): medições, estimativa, predição no escuro e as regiões de busca."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f0, f1 = s - pre, s + n + post
    w = log[(log.frame >= f0) & (log.frame < f1)]
    z, est = to_ref(w, "z_x", "z_y"), to_ref(w, "est_x", "est_y")
    gap = ((w.frame >= s) & (w.frame < s + n)).to_numpy()
    p = path.loc[f0:f1 - 1]
    a = annr[(annr.frame >= f0) & (annr.frame < f1)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.plot(p.rx, p.ry, "-", color="0.75", lw=1, label="bola (marcações interpoladas)")
    ax.plot(a.rx, a.ry, "kx", ms=6, label="marcação")
    ax.plot(z[:, 0], z[:, 1], "o", ms=4, color="tab:blue", label="medição aceita")
    ax.plot(est[~gap, 0], est[~gap, 1], ".", ms=4, color="tab:orange", label="estimativa")
    ax.plot(est[gap, 0], est[gap, 1], ".", ms=5, color="tab:red", label="predição na oclusão")
    for f in range(s, s + n, 5):
        st = steps.get(f)
        if st is not None and st.S is not None:
            e = ellipse(st.z_pred, st.S, threshold)
            ax.plot(*np.vstack([e, e[:1]]).T, "-", color="tab:red", lw=.8, alpha=.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.set_xlabel("x (px, referência)")
    ax.set_ylabel("y (px)")
    ax.set_title("plano da imagem; elipses = região de busca a cada 5 quadros", fontsize=9)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    d = np.hypot(est[:, 0] - path.rx.reindex(w.frame).to_numpy(), est[:, 1] - path.ry.reindex(w.frame).to_numpy())
    ax.plot(w.frame, d, ".-", color="tab:orange", lw=1)
    ax.plot(w.frame[gap], d[gap], ".", color="tab:red")
    ax.axvspan(s - .5, s + n - .5, color="tab:red", alpha=.08, label=f"oclusão ({n} quadros)")
    ax.axhline(10, color="k", ls=":", lw=1)
    ax.set_xlabel("quadro")
    ax.set_ylabel("distância estimativa -> bola (px)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- linha de comando
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--n", type=int, nargs="+", default=[10, 20, 30], help="duração das oclusões (quadros)")
    ap.add_argument("--modes", nargs="+", default=["bola", "tudo"], choices=["bola", "tudo"])
    ap.add_argument("--max-dark", type=int, nargs="+", default=[20])
    ap.add_argument("--remove-radius", type=float, default=20.0, help="px: candidatos apagados em volta da bola")
    ap.add_argument("--sigma-m", type=float, default=TrackerParams.sigma_m)
    args = ap.parse_args()

    stem = Path(args.video).stem
    scorer = args.scorer or f"data/models/ball_scorer_{stem}.json"
    exp_dir = Path("outputs/experiments") / stem
    data = collect(args.video, scorer, cache=exp_dir / "replay_cache.pkl")
    pw = data["size"][0]
    fps = data["fps"]
    ann = load_annotations(Path("data/annotations") / f"{stem}.csv", scale=pw / data["orig_width"])
    path = ball_path(ann, data)
    from experiments.e2_sigma_a import annotations_ref
    annr = annotations_ref(ann, dict(zip(data["frame"], data["H"])))
    field_path = Path("data/calib") / f"{stem}_field.json"
    field = FieldModel(Calibration.load(field_path), pw) if field_path.exists() else None

    filters = {"F1": (TrackerParams(fps=fps, sigma_m=args.sigma_m), False)}
    if field is not None:
        filters["F2"] = (TrackerParams.metric(fps=fps), True)
    base = {name: replay(data, tp, field, metric, args.sigma_m) for name, (tp, metric) in filters.items()}
    # trilhas normais com cada max_dark: os trechos de oclusão têm de ser justos em todas
    base_all = [replay(data, replace(tp, max_dark=md), field, metric, args.sigma_m)
                for md in args.max_dark for tp, metric in filters.values()]

    # ---- oclusões naturais (trilha normal, parâmetros padrão)
    nat = []
    for name, log in base.items():
        df = natural_episodes(log, ann, path, fps)
        df.insert(0, "filtro", name)
        nat.append(df)
    nat = pd.concat(nat, ignore_index=True)
    nat.to_csv(exp_dir / "e3_naturais.csv", index=False)

    # ---- oclusões simuladas
    per_gap, summary, curves, example = [], [], {}, None
    for n in args.n:
        starts = choose_gaps(base_all, path, n)
        print(f"  N={n}: {len(starts)} oclusoes simuladas", flush=True)
        for mode in args.modes:
            occ = occlude(data, starts, n, path, mode, args.remove_radius)
            for md in args.max_dark:
                for name, (tp, metric) in filters.items():
                    tpm = replace(tp, max_dark=md)
                    steps = {}
                    log = replay(occ, tpm, field, metric, args.sigma_m, steps_out=steps)
                    thr = chi2_threshold(tpm.gate_prob)
                    rows = gap_metrics(log, steps, starts, n, path, thr)
                    for r in rows:
                        per_gap.append({"filtro": name, "modo": mode, "max_dark": md,
                                        **{k: v for k, v in r.items() if k != "err_k"}})
                    summary.append({"filtro": name, "modo": mode, "max_dark": md, "N": n, **summarize(rows)})
                    if mode == "bola" and n == max(args.n):
                        curves[f"{name}, max_dark {md}"] = rows
                    if name == "F1" and mode == "bola" and md == args.max_dark[0] and n == 20 and rows:
                        # exemplo: a oclusão em que a bola mais anda, retomada na mesma trilha
                        valid = [r for r in rows if r["valid"]] or rows
                        good = [r for r in valid if r["reacq_same_track"]] or valid
                        disp = [np.hypot(path.rx[r["start"] + n] - path.rx[r["start"]],
                                         path.ry[r["start"] + n] - path.ry[r["start"]]) for r in good]
                        example = (log, steps, good[int(np.argmax(disp))]["start"], thr)

    per_gap = pd.DataFrame(per_gap)
    per_gap.to_csv(exp_dir / "e3_simulada.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(exp_dir / "e3_resumo.csv", index=False)

    # ---- gráficos
    tp1 = filters["F1"][0]
    gate = gate_growth(tp1, max(args.n))
    plot_error_k(curves, gate, tp1.max_gate_px, args.max_dark[0], exp_dir / "e3_erro_k.png",
                 f"E3 {stem}: oclusao simulada de {max(args.n)} quadros (so a bola some)")
    if example is not None:
        log, steps, s, thr = example
        plot_example(log, steps, path, annr, s, 20, thr, exp_dir / f"e3_exemplo_{s}.png",
                     f"E3 {stem} F1: oclusao simulada de 20 quadros a partir do quadro {s}")

    (exp_dir / "e3.json").write_text(json.dumps({
        "summary": summary, "natural": nat.to_dict(orient="records"),
        "gate_growth_F1_px": [round(float(g), 2) for g in gate], "gate_cap_F1_px": tp1.max_gate_px,
        "remove_radius_px": args.remove_radius,
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.to_string(index=False))
        print()
        nb = nat.assign(na_bola_antes=nat.na_bola_antes.fillna("?").astype(str))
        print(nb.groupby(["filtro", "na_bola_antes", "motivo"]).agg(
            episodios=("quadros", "size"), recuperados=("recuperada", "sum"),
            reaceita_na_bola=("reaceita_na_bola", lambda x: int((x == True).sum()))).to_string())  # noqa: E712
    print("raio da regiao de busca F1 (px) em k = 1, 5, 10, 20, 30:",
          [round(float(gate[k - 1]), 1) for k in (1, 5, 10, 20, 30) if k <= len(gate)])
    print(f"saidas em {exp_dir}")


if __name__ == "__main__":
    main()
