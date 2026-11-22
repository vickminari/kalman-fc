"""experiments/skin_check.py: máscara de pele, AUC e estatísticas de cor de um candidato."""
import cv2
import numpy as np

from experiments.skin_check import auc, color_stats, skin_mask


def test_skin_mask_accepts_orange_skin_and_rejects_white_and_grass():
    hsv = np.array([[[15, 80, 200], [0, 10, 230], [40, 130, 110], [120, 20, 220]]], np.uint8)
    assert skin_mask(hsv).tolist() == [[True, False, False, False]]


def test_auc_perfect_and_chance():
    assert auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auc([0, 1], [2, 3]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5


def test_color_stats_on_skin_disc_inside_cool_ring():
    hsv = np.zeros((60, 60, 3), np.uint8)
    hsv[...] = (120, 20, 220)                               # fundo frio
    cv2.circle(hsv, (30, 30), 6, (15, 80, 200), -1)         # disco cor de pele
    st = color_stats(hsv, 30, 30, 12)
    assert st["skin"] > 0.9 and st["cool"] < 0.1 and st["h_med"] == 15
    assert st["skin_ring"] < 0.1
