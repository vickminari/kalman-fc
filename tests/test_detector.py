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
