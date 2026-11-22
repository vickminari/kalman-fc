"""Diagnóstico de uma trilha já rodada: a "régua" da limpeza do filtro (report 010/011).

Lê `outputs/<tag>/track.csv` (+ `track.json` para os parâmetros) e as marcações, e mede:

- **métricas de referência:** "na bola" (teste e vídeo todo), aceitas com a bola
  invisível, erro mediano da predição no escuro (teste), velocidade p95;
- **ganho de Kalman em regime** para os parâmetros usados (F2: σm em metros pelo
  jacobiano da homografia no ponto mediano das medições);
- **distância estimativa -> medição** e **tremor** (2ª diferença) da medição e da estimativa;
- **saltos** > N px da estimativa entre quadros, classificados em:
  trilha nova / no escuro / retomada longe da predição / outro;
- **deslocamento por quadro no escuro** (mediana, p90);
- **associação:** medições aceitas na bola / fora da bola / com a bola invisível;
  trilhas nascidas na bola;
- **episódios no escuro:** quantos terminam com a bola recuperada e quantos perdem a trilha.

Tudo em px de processamento (640 de largura). Saltos, tremor e deslocamentos são medidos,
por padrão, no **quadro de referência** (estabilizado): as colunas h00..h22 do CSV levam o
quadro atual para a referência, então F1 e F2 ficam na mesma unidade e o balanço da câmera
não conta como movimento da bola. Com `--coords cur`, mede no quadro atual, como o script
avulso do report 010 (assim os números dele são reproduzidos: 118 saltos, 62 no escuro,
tremor da medição 0,61 px). "Teste" = quadros marcados a partir da mediana dos quadros marcados (a mesma
divisão do treino do score combinado: lateral >= 2012, centro >= 1820).

Uso (a partir da raiz do repositório):

    python -m experiments.track_diagnostics lateral_F1
    python -m experiments.track_diagnostics lateral_F1 lateral_F2 centro_F1 centro_F2
    python -m experiments.track_diagnostics outputs/lateral_F1/track.csv --jump 20 --coords cur

Salva `diagnostics.json` ao lado de cada `track.csv`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import evaluate_track, load_annotations
from src.kalman import constant_velocity, steady_state_gain

H_COLS = [f"h{i}{j}" for i in range(3) for j in range(3)]


# ---------------------------------------------------------------------- utilidades
def to_ref(log: pd.DataFrame, xcol: str, ycol: str) -> np.ndarray:
    """Pontos do quadro atual -> quadro de referência, quadro a quadro (H de cada linha).

    Linhas sem ponto (NaN) continuam NaN. Sem as colunas de H, devolve os pontos como estão.
    """
    pts = log[[xcol, ycol]].to_numpy(float)
    if not set(H_COLS) <= set(log.columns):
        return pts
    H = log[H_COLS].to_numpy(float).reshape(-1, 3, 3)
    a = np.einsum("nij,nj->ni", H, np.column_stack([pts, np.ones(len(pts))]))
    return a[:, :2] / a[:, 2:3]


def kalman_gain(fps: float, sigma_a: float, Sm: np.ndarray) -> dict:
    """K da posição em regime (autovalores do bloco 2x2: iguais no F1, diferentes no F2)."""
    m = constant_velocity(1.0 / fps, sigma_a, 1.0)
    K, _ = steady_state_gain(m.D, m.M, m.Sd, np.asarray(Sm, float))
    lam = np.sort(np.linalg.eigvals(K[:2, :2]).real)
    return {"K_pos_min": round(float(lam[0]), 3), "K_pos_max": round(float(lam[1]), 3),
            "K_vel_per_s": round(float(np.sort(np.linalg.eigvals(K[2:, :2]).real)[-1]), 2)}


def split_frame(ann: pd.DataFrame) -> int:
    """Primeiro quadro da metade de teste (mediana dos quadros marcados)."""
    return int(np.median(sorted(ann.frame)))


def _stats(d) -> dict | None:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return None
    return {"n": int(len(d)), "median": round(float(np.median(d)), 2),
            "p90": round(float(np.quantile(d, .9)), 2), "max": round(float(d.max()), 2)}


# ---------------------------------------------------------------------- diagnóstico
def jumps(log: pd.DataFrame, est_ref: np.ndarray, jump_px: float) -> dict:
    """Saltos da estimativa entre quadros consecutivos, classificados pela causa.

    Ordem da classificação (a primeira que valer):
      trilha_nova  o quadro inicia uma trilha (o salto liga a trilha velha à nova);
      no_escuro    quadro sem medição aceita: a predição andou sozinha;
      retomada     medição aceita logo depois de quadros no escuro (dark > 0 no anterior):
                   a bola voltou longe de onde a predição estava;
      outro        com medição nos dois quadros (chute de verdade ou troca de alvo).
    """
    frames = log.frame.to_numpy()
    step = np.full(len(log), np.nan)
    ok = np.r_[False, np.diff(frames) == 1]          # quadros consecutivos
    d = np.linalg.norm(np.diff(est_ref, axis=0), axis=1)
    step[1:] = np.where(ok[1:], d, np.nan)
    big = np.flatnonzero(step > jump_px)
    has_z = log.z_x.notna().to_numpy()
    new = log.new_track.astype(bool).to_numpy()
    dark_state = (log.state == "NO_ESCURO").to_numpy()
    dark_prev = np.r_[0, log.dark.to_numpy()[:-1]] > 0
    kinds = {"trilha_nova": 0, "no_escuro": 0, "retomada": 0, "outro": 0}
    for i in big:
        if new[i]:
            kinds["trilha_nova"] += 1
        elif dark_state[i] or not has_z[i]:
            kinds["no_escuro"] += 1
        elif dark_prev[i]:
            kinds["retomada"] += 1
        else:
            kinds["outro"] += 1
    in_dark = step[dark_state]
    return {"threshold_px": jump_px, "total": int(len(big)), **kinds,
            "dark_step_px": _stats(in_dark)}


def dark_episodes(log: pd.DataFrame) -> dict:
    """Sequências de quadros NO_ESCURO: terminam recuperando a bola ou perdendo a trilha."""
    st = (log.state == "NO_ESCURO").to_numpy()
    has_z = log.z_x.notna().to_numpy()
    lengths_rec, lengths_lost = [], []
    i, n = 0, len(st)
    while i < n:
        if not st[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and st[j + 1]:
            j += 1
        L = j - i + 1
        if j + 1 < n and has_z[j + 1] and not bool(log.new_track.iloc[j + 1]):
            lengths_rec.append(L)
        else:
            lengths_lost.append(L)
        i = j + 1
    rec = np.array(lengths_rec)
    return {"episodes": len(lengths_rec) + len(lengths_lost),
            "recovered": len(lengths_rec), "lost": len(lengths_lost),
            "recovered_10_30": int(((rec >= 10) & (rec <= 30)).sum()) if len(rec) else 0,
            "recovered_len": _stats(rec), "lost_len": _stats(lengths_lost)}


def association(log: pd.DataFrame, ann: pd.DataFrame, radius: float) -> dict:
    """Medições aceitas contra as marcações, e onde as trilhas nasceram."""
    m = ann.merge(log, on="frame", how="inner")
    acc = m[m.z_x.notna()]
    vis = acc[acc.visible == 1]
    d = np.hypot(vis.z_x - vis.gx, vis.z_y - vis.gy)
    births = m[m.new_track.astype(bool)]
    bv = births[births.visible == 1]
    born_on = int((np.hypot(bv.est_x - bv.gx, bv.est_y - bv.gy) <= radius).sum())
    visible = m[m.visible == 1]
    return {"accepted_visible": int(len(vis)), "accepted_on_ball": int((d <= radius).sum()),
            "accepted_off_ball": int((d > radius).sum()),
            "accepted_invisible": int((acc.visible == 0).sum()),
            "visible_without_measurement": int(visible.z_x.isna().sum()),
            "visible_frames": int(len(visible)),
            "tracks_started": int(log.new_track.astype(bool).sum()),
            "births_marked": int(len(births)), "births_on_ball": born_on}


def diagnose(log: pd.DataFrame, ann: pd.DataFrame, fps: float, sigma_a: float,
             Sm: np.ndarray, split: int | None = None, jump_px: float = 20.0,
             radius: float = 10.0, coords: str = "ref") -> dict:
    """Todas as medidas para um log do `src.main` (ver docstring do módulo).

    `coords`: "ref" (quadro de referência, padrão) ou "cur" (quadro atual) para saltos e tremor.
    """
    log = log.sort_values("frame").reset_index(drop=True)
    split = split_frame(ann) if split is None else split
    if coords == "ref":
        est_ref = to_ref(log, "est_x", "est_y")
        z_ref = to_ref(log, "z_x", "z_y")
    else:
        est_ref = log[["est_x", "est_y"]].to_numpy(float)
        z_ref = log[["z_x", "z_y"]].to_numpy(float)

    ev_all = evaluate_track(log, ann, radius)
    ev_test = evaluate_track(log[log.frame >= split], ann[ann.frame >= split], radius)
    ref = {
        "on_ball_test_pct": ev_test.get("estimate_on_ball_pct"),
        "on_ball_all_pct": ev_all.get("estimate_on_ball_pct"),
        "accepted_invisible_all": ev_all.get("accepted_while_invisible"),
        "dark_pred_err_test_median_px": (ev_test.get("error_dark_prediction_px") or {}).get("median"),
        "dark_pred_err_all_median_px": (ev_all.get("error_dark_prediction_px") or {}).get("median"),
    }
    if "speed_kmh" in log:
        sp = log.speed_kmh.dropna()
        ref["speed_kmh_p95"] = round(float(sp.quantile(.95)), 1) if len(sp) else None

    # estimativa x medição e tremor, só nos quadros com medição aceita
    has_z = log.z_x.notna().to_numpy()
    gap = np.linalg.norm(est_ref[has_z] - z_ref[has_z], axis=1)
    consec = np.r_[False, np.diff(log.frame.to_numpy()) == 1]
    run3 = has_z & np.r_[False, has_z[:-1]] & np.r_[False, False, has_z[:-2]] \
        & consec & np.r_[False, consec[:-1]]

    def tremor(p):
        dd = p[2:] - 2 * p[1:-1] + p[:-2]
        return np.linalg.norm(dd, axis=1)[run3[2:]]

    states = log.state.value_counts(normalize=True).mul(100).round(1).to_dict()
    return {
        "frames": int(len(log)), "split_frame": split, "coords": coords, "states_pct": states,
        "reference": ref,
        "kalman_gain": kalman_gain(fps, sigma_a, Sm),
        "est_to_meas_px": _stats(gap),
        "tremor_px": {"measurement": _stats(tremor(z_ref)), "estimate": _stats(tremor(est_ref))},
        "jumps": jumps(log, est_ref, jump_px),
        "dark_episodes": dark_episodes(log),
        "association": association(log, ann, radius),
    }


# ---------------------------------------------------------------------- linha de comando
def _resolve(arg: str) -> Path:
    p = Path(arg)
    if p.suffix == ".csv":
        return p
    if p.is_dir():
        return p / "track.csv"
    return Path("outputs") / arg / "track.csv"


def typical_Sm(log: pd.DataFrame, sigma_m_px: float, field=None) -> np.ndarray:
    """Σm típico, nas unidades do filtro, para o K em regime.

    F1 (field=None): σm² I, em px. F2: Σm em metros = J Σm_px Jᵀ, com o jacobiano da
    homografia no ponto mediano das medições aceitas (no quadro de referência).
    """
    Sm = np.eye(2) * sigma_m_px ** 2
    if field is None:
        return Sm
    uv = np.nanmedian(to_ref(log, "z_x", "z_y"), axis=0)
    J = field.jacobian(uv)
    return J @ Sm @ J.T


def run_one(csv: Path, jump_px: float, ann_arg: str | None, coords: str = "ref") -> dict:
    meta = json.loads((csv.parent / "track.json").read_text(encoding="utf-8"))
    log = pd.read_csv(csv)
    pw, _ = meta["processing_size"]
    video = Path(meta["video"])
    orig_w = 1920   # vídeos da dupla; as marcações são no tamanho original
    ann_path = Path(ann_arg) if ann_arg else Path("data/annotations") / f"{video.stem}.csv"
    ann = load_annotations(ann_path, scale=pw / orig_w)
    fps = 1.0 / float(np.median(np.diff(log.t)))
    prm = meta["params"]
    field = None
    if prm["filter"].startswith("F2"):
        from src.field import Calibration, FieldModel
        field = FieldModel(Calibration.load(Path(prm["field"].replace("\\", "/"))), pw)
    Sm = typical_Sm(log, prm.get("sigma_m_px", 1.5), field)
    out = diagnose(log, ann, fps, prm["sigma_a"], Sm, jump_px=jump_px, coords=coords)
    out["source"] = str(csv)
    out["params"] = prm
    (csv.parent / "diagnostics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    return out


def _row(tag: str, r: dict) -> dict:
    ref, j, a, de = r["reference"], r["jumps"], r["association"], r["dark_episodes"]
    tr = r["tremor_px"]
    g = r["est_to_meas_px"] or {}
    return {
        "tag": tag,
        "bola_teste%": ref["on_ball_test_pct"], "bola_todo%": ref["on_ball_all_pct"],
        "aceit_invis": ref["accepted_invisible_all"],
        "escuro_err_teste": ref["dark_pred_err_test_median_px"],
        "v_p95_kmh": ref.get("speed_kmh_p95"),
        "K_pos": f"{r['kalman_gain']['K_pos_min']:.2f}-{r['kalman_gain']['K_pos_max']:.2f}",
        "est-med": g.get("median"),
        "tremor_m/e": f"{(tr['measurement'] or {}).get('median')}/{(tr['estimate'] or {}).get('median')}",
        "saltos": j["total"], "s_escuro": j["no_escuro"], "s_retom": j["retomada"],
        "s_nova": j["trilha_nova"], "s_outro": j["outro"],
        "escuro_px/q": f"{(j['dark_step_px'] or {}).get('median')}/{(j['dark_step_px'] or {}).get('p90')}",
        "aceit_bola": f"{a['accepted_on_ball']}/{a['accepted_visible']}",
        "trilhas": a["tracks_started"],
        "escuro_rec/perd": f"{de['recovered']}/{de['lost']}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", nargs="+", help="tag em outputs/, pasta ou caminho do track.csv")
    ap.add_argument("--jump", type=float, default=20.0, help="limiar de salto entre quadros (px)")
    ap.add_argument("--coords", choices=["ref", "cur"], default="ref",
                    help="saltos e tremor no quadro de referência (padrão) ou no atual")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--out", default=None, help="CSV com a tabela comparativa")
    args = ap.parse_args()

    rows = []
    for t in args.tracks:
        csv = _resolve(t)
        r = run_one(csv, args.jump, args.annotations, args.coords)
        rows.append(_row(csv.parent.name, r))
        print(f"{csv}: diagnostics.json salvo")
    table = pd.DataFrame(rows).set_index("tag").T
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string())
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
