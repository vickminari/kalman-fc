"""Estimativa do movimento global da câmera entre quadros consecutivos.

Serve para checar se a câmera ficou realmente fixa (requisito para MOG2 e homografia)
e, se necessário, para compensar pequenos balanços.
"""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotion:
    """Deslocamento (dx, dy) entre quadros consecutivos por correlação de fase."""

    def __init__(self, work_width: int = 320):
        self.work_width = work_width
        self._prev: np.ndarray | None = None
        self._win: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray) -> tuple[float, float, float]:
        """Retorna (dx, dy, resposta) em pixels do quadro recebido. Primeiro quadro: zeros."""
        h, w = frame_bgr.shape[:2]
        s = self.work_width / w
        small = cv2.resize(frame_bgr, (self.work_width, int(round(h * s))), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._win is None or self._win.shape != g.shape:
            self._win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if self._prev is None:
            self._prev = g
            return 0.0, 0.0, 1.0
        (dx, dy), resp = cv2.phaseCorrelate(self._prev, g, self._win)
        self._prev = g
        return dx / s, dy / s, float(resp)
