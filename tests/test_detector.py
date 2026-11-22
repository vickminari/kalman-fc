"""Testes do detector com uma cena sintética: parede branca em cima, grama embaixo."""
import cv2
import numpy as np

from src.detector import BallDetector


def make_scene(w=640, h=362, field_top=120):
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:field_top] = (15, 25, 200)        # parede clara
    hsv[field_top:] = (38, 130, 110)       # grama
    rng = np.random.default_rng(0)
    noise = rng.integers(-8, 9, size=(h - field_top, w), dtype=np.int16)
    hsv[field_top:, :, 2] = np.clip(110 + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_detects_white_ball_on_grass():
    img = make_scene()
    cv2.circle(img, (300, 250), 7, (235, 235, 235), -1)
    det = BallDetector().detect(img)
    assert det.best is not None
    assert abs(det.best.x - 300) < 2 and abs(det.best.y - 250) < 2
    assert 11 < det.best.diameter < 17


def test_ball_with_colored_details_is_one_candidate():
    img = make_scene()
    cv2.circle(img, (200, 300), 9, (235, 235, 235), -1)
    cv2.circle(img, (197, 298), 3, (40, 40, 200), -1)   # detalhe vermelho
    det = BallDetector().detect(img)
    near = [c for c in det.candidates if abs(c.x - 200) < 5 and abs(c.y - 300) < 5]
    assert len(near) == 1


def test_rejects_elongated_and_wall_blobs():
    img = make_scene()
    cv2.rectangle(img, (400, 200), (404, 240), (235, 235, 235), -1)  # "canela" alongada
    cv2.circle(img, (100, 60), 7, (250, 250, 250), -1)               # mancha na parede
    det = BallDetector().detect(img)
    assert det.candidates == []


def test_field_border_ignores_player_in_front():
    img = make_scene(field_top=120)
    cv2.rectangle(img, (300, 60), (340, 200), (60, 60, 60), -1)      # jogador escuro sobre a borda
    det = BallDetector().detect(img)
    assert abs(float(det.field_top[320]) - 120) <= 3


# ------------------------------------------------------------------ cor da bola (report 015)
def test_cool_fraction_counts_cool_hue_pixels_in_disc():
    from src.detector import cool_fraction
    hsv = np.zeros((40, 40, 3), np.uint8)
    hsv[..., 0], hsv[..., 2] = 15, 200            # matiz quente (pele)
    assert cool_fraction(hsv, 20, 20, 6) == 0.0
    hsv[..., 0] = 120                             # azul, mesmo com saturação baixa
    assert cool_fraction(hsv, 20, 20, 6) == 1.0
    hsv[:, :20, 0] = 15                           # metade esquerda quente
    assert 0.35 < cool_fraction(hsv, 20, 20, 6) < 0.65
    hsv[..., 2] = 30                              # escuro demais: não conta
    assert cool_fraction(hsv, 20, 20, 6) == 0.0


def test_detector_measures_cool_hue_of_bluish_ball_and_not_of_skin_blob():
    img = make_scene()
    cv2.circle(img, (200, 250), 8, (250, 232, 222), -1)    # branco azulado (BGR), como sob os refletores
    cv2.circle(img, (420, 250), 8, (200, 215, 240), -1)    # "pé": claro e alaranjado
    det = BallDetector().detect(img)
    ball = [c for c in det.candidates if abs(c.x - 200) < 3]
    skin = [c for c in det.candidates if abs(c.x - 420) < 3]
    assert ball and ball[0].cool > 0.8
    assert not skin or skin[0].cool < 0.1


def test_scorer_reads_cool_attribute():
    from src.detector import Candidate
    from src.scoring import BallScorer
    sc = BallScorer(features=["cool"], mean=[0.0], std=[1.0], weights=[0.0, 1.0], fill=[0.0],
                    threshold=0.0, size_prior=[0.0, 0.0], lss_template=[])
    cands = [Candidate(1, 1, 10, 3, 3, 1, 1, 1, cool=0.7), Candidate(1, 1, 10, 3, 3, 1, 1, 1)]
    X = sc.feature_matrix(cands, np.zeros(2), None, 640)
    assert np.allclose(X[:, 0], [0.7, 0.0])
