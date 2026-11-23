"""experiments/e1_raw_vs_filtered.py e experiments/e3_occlusion.py em cenários sintéticos."""
import numpy as np
import pandas as pd

from experiments.e1_raw_vs_filtered import ball_path, jump_count, top_candidate, tremor, _runs3
from experiments.e3_occlusion import (choose_gaps, gap_metrics, gate_growth, natural_episodes,
                                      occlude, summarize)
from experiments.replay_tracker import replay
from src.gating import chi2_threshold
from src.tracker import TrackerParams


def straight_ball(n=400, v=2.0, false_pt=(500.0, 60.0), shift=(0.0, 0.0)):
    """Bola em linha reta a `v` px/quadro, um candidato falso parado longe, câmera que translada
    `shift` (H: atual -> referência). Devolve (cache no formato de `collect`, marcações)."""
    f = np.arange(n)
    ref = np.column_stack([60 + v * f, 200 + 0.0 * f])
    H = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1.0]])
    cur = ref - np.array(shift)
    cands = [np.array([cur[i], false_pt]) for i in range(n)]
    scores = [np.array([0.9, 0.4]) for _ in range(n)]
    data = dict(fps=30.0, frame=list(range(n)), H=[H] * n, cands=cands, scores=scores,
                size=(640, 360), orig_width=640)
    marks = f[::5]
    ann = pd.DataFrame(dict(frame=marks, t=marks / 30.0, x=cur[marks, 0], y=cur[marks, 1], visible=1))
    ann["gx"], ann["gy"] = ann.x, ann.y
    return data, ann


# ---------------------------------------------------------------------- E1
def test_top_candidate_takes_highest_score_and_nan_without_candidates():
    data = dict(frame=[0, 1], cands=[np.array([[1.0, 2.0], [3.0, 4.0]]), np.zeros((0, 2))],
                scores=[np.array([0.2, 0.7]), np.zeros(0)])
    raw = top_candidate(data)
    assert raw.loc[0, ["raw_x", "raw_y"]].tolist() == [3.0, 4.0]
    assert raw.loc[1, ["raw_x", "raw_y"]].isna().all()


def test_ball_path_interpolates_and_returns_to_current_frame():
    data, ann = straight_ball(n=60, shift=(5.0, -3.0))
    ann.loc[ann.frame == 30, "visible"] = 0          # bola invisível no quadro 30
    p = ball_path(ann, data)
    # entre 10 e 15 (as duas visíveis): interpolada; o quadro atual é a referência - shift
    assert np.allclose(p.loc[12, ["rx", "ry"]], [60 + 2 * 12, 200])
    assert np.allclose(p.loc[12, ["cx", "cy"]], [60 + 2 * 12 - 5, 203])
    # de 25 a 35 passa por uma marcação invisível: sem posição
    assert p.loc[27:33, "rx"].isna().all()


def test_tremor_and_jumps():
    frames = np.arange(6)
    p = np.column_stack([np.arange(6) * 2.0, np.zeros(6)])
    p[3, 1] = 1.0                                     # um desvio de 1 px no quadro 3
    mask = _runs3(frames, np.ones(6, bool))
    t = tremor(p, mask)
    assert np.allclose(t, [0.0, 1.0, 2.0, 1.0])     # 2ª diferença em y: 0, 1, -2, 1
    q = p.copy()
    q[4] = [100.0, 100.0]                             # troca de alvo: dois saltos (ida e volta)
    assert jump_count(frames, q, 20.0) == 2


# ---------------------------------------------------------------------- E3
def test_occlude_removes_only_the_ball_or_everything_inside_the_gap():
    data, ann = straight_ball(n=60)
    p = ball_path(ann, data)
    occ = occlude(data, [20], 5, p, "bola", radius=10.0)
    assert len(occ["cands"][19]) == 2 and len(occ["cands"][25]) == 2      # fora do buraco
    for f in range(20, 25):
        assert len(occ["cands"][f]) == 1 and np.allclose(occ["cands"][f][0], [500, 60])
        assert len(occ["scores"][f]) == 1
    tudo = occlude(data, [20], 5, p, "tudo")
    assert all(len(tudo["cands"][f]) == 0 for f in range(20, 25))
    assert len(data["cands"][22]) == 2                                    # o original não muda


def test_gate_grows_with_covariance_and_faster_for_nervous_filter():
    g = gate_growth(TrackerParams(sigma_a=2000), 10)
    assert np.all(np.diff(g) > 0)
    g_stubborn = gate_growth(TrackerParams(sigma_a=300), 10)
    assert g_stubborn[-1] < g[-1]


def test_simulated_occlusion_end_to_end():
    data, ann = straight_ball()
    p = ball_path(ann, data)
    tp = TrackerParams(fps=30.0)
    base = replay(data, tp, None, False)
    starts = choose_gaps([base], p, 10)
    assert len(starts) >= 3
    thr = chi2_threshold(tp.gate_prob)

    # 10 quadros (< max_dark): a predição segue a linha reta e a MESMA trilha retoma a bola
    steps = {}
    log = replay(occlude(data, starts, 10, p, "bola"), tp, None, False, steps_out=steps)
    rows = gap_metrics(log, steps, starts, 10, p, thr)
    assert all(r["kept"] and r["covered_10"] == 10 and r["wrong_accepts"] == 0 for r in rows)
    assert all(r["reacq_frames"] == 0 and r["reacq_same_track"] for r in rows)
    assert all(r["gate_end"] > 10 for r in rows)                          # a região cresceu
    s = summarize(rows)
    assert s["mantida%"] == 100 and s["recup_10px%"] == 100 and s["reaceita_mesma_trilha%"] == 100

    # 30 quadros (> max_dark = 20), apagão: a trilha é perdida no 21º quadro e a bola volta
    # numa trilha nova
    starts30 = choose_gaps([base], p, 30)
    log = replay(occlude(data, starts30, 30, p, "tudo"), tp, None, False)
    rows = gap_metrics(log, {}, starts30, 30, p, thr)
    assert all(r["valid"] for r in rows)
    assert all(not r["kept"] and r["lost_at_k"] == tp.max_dark + 1 for r in rows)
    assert all(r["reacq_frames"] is not None and not r["reacq_same_track"] for r in rows)

    # só a bola some por 30 quadros: perdida a trilha, o falso parado vira trilha nova e prende
    # o filtro; a oclusão seguinte começa com a trilha fora da bola e não entra no resumo
    log = replay(occlude(data, starts30, 30, p, "bola"), tp, None, False)
    rows = gap_metrics(log, {}, starts30, 30, p, thr)
    assert rows[0]["valid"] and not rows[1]["valid"]
    assert summarize(rows)["invalidas"] >= 1


def test_natural_episodes_report_reason_and_recovery():
    data, ann = straight_ball(n=200)
    p = ball_path(ann, data)
    for f in range(100, 112):                         # o detector perde a bola por 12 quadros
        data["cands"][f] = data["cands"][f][1:]
        data["scores"][f] = data["scores"][f][1:]
    log = replay(data, TrackerParams(fps=30.0), None, False)
    nat = natural_episodes(log, ann, p, 30.0)
    assert len(nat) == 1
    e = nat.iloc[0]
    assert e.inicio == 100 and e.quadros == 12 and e.motivo.startswith("bola visivel")
    assert e.na_bola_antes and e.recuperada and e.reaceita_na_bola
    assert e.erro_pred_mediano_px < 1.0
