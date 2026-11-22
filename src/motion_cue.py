"""Pista de movimento por candidato, com a câmera compensada.

Folhas, papéis e a base da rede ficam PARADOS no mundo; a bola em jogo se move. Como a
câmera está na mão, não dá para comparar pixels na mesma posição da imagem. Usamos as
homografias da estabilização para achar, no quadro t−k, o ponto do MUNDO que hoje está
no candidato:

    p_(t−k) = H_(t−k)⁻¹ · H_t · p_t        (H: quadro -> referência)

Depois comparamos o patch em volta do candidato no quadro t com o patch em volta de
p_(t−k) no quadro t−k, permitindo um pequeno deslocamento (±`search` px) para absorver o
erro residual da estabilização (~2 px).

    movimento = mínimo, sobre os deslocamentos, da diferença absoluta média (níveis de cinza)

Objeto parado: o patch casa e o movimento fica baixo. Bola rolando: no quadro t−k aquele
ponto era grama, e o movimento fica alto. Bola parada também dá movimento baixo, então
isto é um ATRIBUTO para o score, não um filtro rígido.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class MotionCue:
    def __init__(self, gap: int = 3, search: int = 2):
        self.gap = gap
        self.search = search
        self._hist: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=gap + 1)

    def update(self, frame_bgr: np.ndarray, H: np.ndarray) -> None:
        """Guarda o quadro atual (cinza) e sua homografia quadro -> referência."""
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self._hist.append((g, H.copy()))

    @property
    def ready(self) -> bool:
        return len(self._hist) > self.gap

    def scores(self, xs: np.ndarray, ys: np.ndarray, diameters: np.ndarray) -> np.ndarray:
        """Movimento de cada candidato do quadro mais recente (nan se não há histórico)."""
        n = len(xs)
        out = np.full(n, np.nan)
        if not self.ready or n == 0:
            return out
        cur, Hc = self._hist[-1]
        old, Ho = self._hist[0]
        M = np.linalg.inv(Ho) @ Hc                     # atual -> quadro t−k
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.float64)
        prev = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        h, w = cur.shape
        s = self.search
        for i in range(n):
            r = int(max(3, round(0.7 * diameters[i])))
            x, y = int(round(xs[i])), int(round(ys[i]))
            px, py = int(round(prev[i, 0])), int(round(prev[i, 1]))
            if not (r <= x < w - r and r <= y < h - r and r + s <= px < w - r - s and r + s <= py < h - r - s):
                continue
            a = cur[y - r:y + r + 1, x - r:x + r + 1]
            b = old[py - r - s:py + r + s + 1, px - r - s:px + r + s + 1]
            # diferença quadrática média para cada deslocamento; fica o mínimo
            ssd = cv2.matchTemplate(b, a, cv2.TM_SQDIFF) / a.size
            out[i] = float(np.sqrt(ssd.min()))
        return out
