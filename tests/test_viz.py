"""Visualização honesta da predição no escuro (src/viz.py e src/minimap.py)."""
import numpy as np

from src.tracker import BallTracker, TrackerParams
from src.viz import Overlay, dashed_polyline, trail_segments


def P(x, y=0.0):
    return np.array([float(x), float(y)])


def test_solid_trail_only_links_measured_frames():
    trail = [(P(0), True), (P(1), True), (P(2), True),        # medição
             (P(10), False), (P(20), False),                   # no escuro
             (P(50), True), (P(51), True)]                     # retomada longe
    solid, dashed = trail_segments(trail, max_drawn_dark=10, keep_past_dark=True)
    assert [len(s) for s in solid] == [3, 2]
    # a retomada não é ligada ao trecho anterior por linha sólida
    assert all(np.linalg.norm(b - a) <= 1.0 for s in solid for a, b in zip(s[:-1], s[1:]))
    # o tracejado começa no último ponto medido e segue a predição
    assert len(dashed) == 1 and np.allclose(dashed[0][0], P(2)) and np.allclose(dashed[0][-1], P(20))
    # por padrão, o trecho no escuro que já terminou (a bola voltou) não é mais desenhado
    assert trail_segments(trail, max_drawn_dark=10)[1] == []


def test_ongoing_dark_stretch_is_drawn():
    trail = [(P(0), True), (P(1), True), (P(10), False), (P(20), False)]
    _, dashed = trail_segments(trail, max_drawn_dark=10)
    assert len(dashed) == 1 and len(dashed[0]) == 3


def test_dark_prediction_is_cut_after_max_drawn_dark():
    trail = [(P(0), True), (P(1), True)] + [(P(10 * k), False) for k in range(1, 31)]
    _, dashed = trail_segments(trail, max_drawn_dark=10)
    assert len(dashed) == 1 and len(dashed[0]) == 1 + 10      # último medido + 10 predições


def test_new_track_separator_breaks_both_kinds_of_segment():
    trail = [(P(0), True), (P(1), True), (P(5), False), None, (P(100), True), (P(101), True)]
    solid, dashed = trail_segments(trail, max_drawn_dark=10, keep_past_dark=True)
    assert len(solid) == 2 and len(dashed) == 1
    assert np.allclose(solid[1][0], P(100))


def test_dashed_polyline_leaves_gaps():
    img = np.zeros((20, 100, 3), np.uint8)
    dashed_polyline(img, np.array([[5.0, 10.0], [95.0, 10.0]]), (255, 255, 255), dash=5, gap=5)
    row = img[10, 5:95, 0] > 0
    assert row.any() and (~row).any()                         # há traços e vãos
    assert 0.3 < row.mean() < 0.8


def test_minimap_dark_prediction_faded_then_hidden():
    from src.field import FieldModel, calibrate
    from src.minimap import Minimap
    from tests.test_field import synthetic_setup
    lm, K, _ = synthetic_setup()
    mm = Minimap(FieldModel(calibrate(lm, K), 640), 360, (640, 360), max_drawn_dark=3)
    base = mm.draw(np.array([6.0, 3.0]), np.array([6.0, 3.0]), 5.0)
    mm.draw(np.array([6.2, 3.0]), np.array([6.2, 3.0]), 5.0)
    imgs = [mm.draw(np.array([6.2 + 0.5 * k, 3.0]), None, 5.0, dark=True) for k in range(1, 7)]
    assert mm.n_dark == 6
    # cada um dos 3 primeiros quadros no escuro acrescenta um ponto; depois o desenho para
    diff = [int((a != b).any(axis=2).sum()) for a, b in zip(imgs[:-1], imgs[1:])]
    assert diff[0] > 0 and diff[-1] < diff[0]
    assert (imgs[-1] != base).any()


def test_overlay_hides_prediction_after_long_dark():
    """Bola andando, depois 25 quadros sem medição: a partir do 11º quadro no escuro, o
    rastro tracejado para de crescer e a estimativa não é mais desenhada."""
    tr = BallTracker(TrackerParams(sigma_a=4000, max_dark=40))
    ov = Overlay(max_drawn_dark=10)
    frame = np.zeros((360, 640, 3), np.uint8)
    H = np.eye(3)
    painted = []
    for t in range(40):
        c = np.array([[100.0 + 5 * t, 180.0]]) if t < 15 else np.zeros((0, 2))
        st = tr.step(c)
        out = ov.draw(frame, st, H, c, 30.0)
        # pixels coloridos fora da faixa do topo e da faixa "NO ESCURO" de baixo
        painted.append(int((out[25:320].sum(axis=2) > 0).sum()))
    # durante a predição desenhada, o desenho cresce; depois do corte, só a elipse muda
    assert painted[15 + 9] > painted[15]
    late = [p for p in painted[15 + 12:]]
    assert max(late) - min(late) < 0.5 * painted[15 + 9]
