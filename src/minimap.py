"""Campo virtual (minimapa) visto de cima, no estilo das transmissões de TV.

Desenha a metade filmada do campo em metros: linha do gol, traves, laterais, a posição
da câmera, a região do chão visível no vídeo e, a cada quadro, a bola (estimativa em
laranja, medição em azul), o rastro e a velocidade em km/h.

Orientação: gol em cima, campo descendo; X cresce para a ESQUERDA, como visto de dentro
do campo olhando para o gol (o sistema do mundo é destro, com Z para cima).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from src.field import GOAL_LINE_LENGTH, POSTS_X, FieldModel
from src.viz import TrailItem, blend, trail_segments

GRASS = (60, 140, 60)
GRASS_DARK = (50, 120, 50)
LINE = (235, 235, 235)
ORANGE = (0, 165, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Minimap:
    def __init__(self, field: FieldModel, height_px: int, proc_size: tuple[int, int],
                 depth_m: float | None = None, margin_m: float = 1.5, trail: int = 90,
                 max_drawn_dark: int = 10, dark_alpha: float = 0.45):
        self.field = field
        cam = field.calib.camera_center
        self.depth = depth_m or float(np.clip(max(cam[1] + 1.0, 10.0), 10.0, 20.0))
        self.x0, self.x1 = -margin_m, GOAL_LINE_LENGTH + margin_m
        self.y0, self.y1 = -2.5, self.depth
        self.h = height_px
        self.ppm = height_px / (self.y1 - self.y0)
        self.w = int(round((self.x1 - self.x0) * self.ppm))
        self.trail: deque[TrailItem] = deque(maxlen=trail)
        self.max_drawn_dark = max_drawn_dark   # como no vídeo: predição no escuro só por N quadros
        self.dark_alpha = dark_alpha
        self.n_dark = 0                        # quadros seguidos sem medição
        self.base = self._background(proc_size)

    # ------------------------------------------------------------------ coordenadas
    def px(self, XY) -> np.ndarray:
        """Mundo (m) -> pixel do minimapa (X cresce para a esquerda)."""
        XY = np.atleast_2d(np.asarray(XY, float))
        u = (self.x1 - XY[:, 0]) * self.ppm
        v = (XY[:, 1] - self.y0) * self.ppm
        return np.stack([u, v], 1)

    def _ipx(self, XY) -> tuple[int, int]:
        p = self.px(XY)[0]
        return int(round(p[0])), int(round(p[1]))

    # ------------------------------------------------------------------ fundo
    def _background(self, proc_size) -> np.ndarray:
        img = np.full((self.h, self.w, 3), GRASS_DARK, np.uint8)
        # faixas de grama a cada 2 m
        for k, y in enumerate(np.arange(0, self.y1, 2.0)):
            col = GRASS if k % 2 == 0 else GRASS_DARK
            a, b = self.px([[GOAL_LINE_LENGTH, y]])[0], self.px([[0, min(y + 2, self.y1)]])[0]
            cv2.rectangle(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, -1)
        # região do chão visível no quadro de referência
        vis = self.field.visible_region(*proc_size)
        if len(vis) >= 3:
            hull = cv2.convexHull(self.px(vis).astype(np.int32))
            over = img.copy()
            cv2.fillPoly(over, [hull], (90, 175, 90))
            img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
        # linhas: gol e laterais
        cv2.line(img, self._ipx([0, 0]), self._ipx([GOAL_LINE_LENGTH, 0]), LINE, 2)
        cv2.line(img, self._ipx([0, 0]), self._ipx([0, self.y1]), LINE, 2)
        cv2.line(img, self._ipx([GOAL_LINE_LENGTH, 0]), self._ipx([GOAL_LINE_LENGTH, self.y1]), LINE, 2)
        # gol (traves e a "caixa" da rede atrás da linha)
        a, b = self._ipx([POSTS_X[1], 0]), self._ipx([POSTS_X[0], -1.0])
        cv2.rectangle(img, a, b, (200, 200, 200), 1)
        for X in POSTS_X:
            cv2.circle(img, self._ipx([X, 0]), 4, WHITE, -1)
        # câmera
        C = self.field.calib.camera_center
        c = self._ipx(C[:2])
        if 0 <= c[0] < self.w and 0 <= c[1] < self.h:
            cv2.drawMarker(img, c, BLACK, cv2.MARKER_TRIANGLE_UP, 14, 2)
            cv2.putText(img, "camera", (c[0] + 8, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 1, cv2.LINE_AA)
        # escala de 1 m
        y = self.h - 10
        cv2.line(img, (10, y), (10 + int(self.ppm), y), WHITE, 2)
        cv2.putText(img, "1 m", (14 + int(self.ppm), y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------ quadro
    def draw(self, est_world: np.ndarray | None, meas_world: np.ndarray | None,
             speed_kmh: float | None, new_track: bool = False, dark: bool = False) -> np.ndarray:
        """Um quadro do minimapa. Como no vídeo (src/viz.py): rastro sólido só com medição;
        sem medição, a predição vira pontos esmaecidos, e some depois de `max_drawn_dark`."""
        img = self.base.copy()
        measured = meas_world is not None
        if new_track:
            self.trail.append(None)
        self.n_dark = 0 if (measured or new_track) else self.n_dark + 1
        self.trail.append(None if est_world is None else (np.asarray(est_world, float).copy(), measured))
        solid, dashed = trail_segments(self.trail, self.max_drawn_dark)
        for seg in solid:
            cv2.polylines(img, [self.px(np.array(seg)).astype(np.int32)], False, ORANGE, 2, cv2.LINE_AA)
        show_pred = est_world is not None and not measured and self.n_dark <= self.max_drawn_dark
        if dashed or show_pred:
            layer = np.zeros_like(img)
            for seg in dashed:
                for q in self.px(np.array(seg[1:])):          # pontos da predição (sem o último medido)
                    cv2.circle(layer, (int(round(q[0])), int(round(q[1]))), 2, ORANGE, -1, cv2.LINE_AA)
            if show_pred:
                cv2.circle(layer, self._ipx(est_world), 5, RED if dark else ORANGE, 1, cv2.LINE_AA)
            blend(img, layer, self.dark_alpha)
        if measured:
            cv2.circle(img, self._ipx(meas_world), 6, BLUE, 2, cv2.LINE_AA)
        if est_world is not None and measured:
            cv2.circle(img, self._ipx(est_world), 5, ORANGE, -1, cv2.LINE_AA)
            cv2.circle(img, self._ipx(est_world), 5, BLACK, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (self.w, 22), BLACK, -1)
        txt = "campo virtual"
        if est_world is not None:
            txt += f"  X={est_world[0]:5.1f} Y={est_world[1]:5.1f} m"
        if speed_kmh is not None:
            txt += f"  {speed_kmh:4.1f} km/h"
        cv2.putText(img, txt, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
        return img
