"""Sobreposições do vídeo de saída, com as cores da aula:

    medição               azul
    predição              vermelho  (+ elipse da região de busca)
    estimativa corrigida  laranja   (+ rastro da trajetória)
    "NO ESCURO"           faixa vermelha quando não há medição aceita

O filtro roda nas coordenadas do quadro de referência (estabilizado, F1) ou do campo em
metros (F2). Para desenhar no quadro atual, os pontos vão para a referência (`to_ref`,
identidade no F1; homografia campo -> imagem no F2) e voltam pela inversa de H
(atual -> referência).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from src.gating import ellipse
from src.stabilize import Stabilizer
from src.tracker import TrackState, TrackStep

BLUE = (255, 0, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
GRAY = (190, 190, 190)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


class Overlay:
    def __init__(self, trail: int = 45, gate_threshold: float = 9.21):
        self.trail: deque[np.ndarray | None] = deque(maxlen=trail)
        self.gate_threshold = gate_threshold

    def draw(self, frame: np.ndarray, step: TrackStep, H: np.ndarray,
             candidates_cur: np.ndarray, fps: float, to_ref=None,
             speed_text: str | None = None) -> np.ndarray:
        """`H`: atual -> referência. `candidates_cur`: candidatos em px do quadro atual.
        `to_ref`: coordenadas do filtro -> px da referência (None = já são px da referência)."""
        out = frame.copy()
        Hinv = np.linalg.inv(H)
        ref = (lambda p: np.asarray(p, float).reshape(-1, 2)) if to_ref is None else to_ref

        def to_cur(pts):
            return Stabilizer.apply(Hinv, ref(np.asarray(pts, float).reshape(-1, 2)))

        for c in candidates_cur:
            cv2.circle(out, _pt(c), 2, GRAY, -1, cv2.LINE_AA)

        # rastro (estimativas passadas, guardadas na referência -> não "andam" com a câmera)
        # uma trilha nova começa um segmento novo (não liga trilhas diferentes)
        if step.new_track:
            self.trail.append(None)
        self.trail.append(None if step.x is None else step.x[:2].copy())
        segment: list[np.ndarray] = []
        for p in list(self.trail) + [None]:
            if p is not None:
                segment.append(p)
                continue
            if len(segment) >= 2:
                cur = to_cur(np.array(segment)).astype(np.int32)
                cv2.polylines(out, [cur], False, ORANGE, 1, cv2.LINE_AA)
            segment = []

        if step.z_pred is not None and step.S is not None:
            ell = to_cur(ellipse(step.z_pred, step.S, self.gate_threshold)).astype(np.int32)
            cv2.polylines(out, [ell], True, RED, 1, cv2.LINE_AA)
            cv2.drawMarker(out, _pt(to_cur(step.z_pred)[0]), RED, cv2.MARKER_CROSS, 10, 2)

        if step.z_index is not None and len(candidates_cur):
            cv2.circle(out, _pt(candidates_cur[step.z_index]), 7, BLUE, 2, cv2.LINE_AA)

        if step.x is not None:
            cv2.circle(out, _pt(to_cur(step.x[:2])[0]), 4, ORANGE, -1, cv2.LINE_AA)

        self._banner(out, step, fps, speed_text)
        return out

    def _banner(self, out: np.ndarray, step: TrackStep, fps: float, speed_text: str | None) -> None:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 20), BLACK, -1)
        if speed_text is not None:
            speed = f"  {speed_text}"
        else:
            speed = "" if step.x is None else f"  v={np.hypot(step.x[2], step.x[3]):5.0f} px/s"
        txt = (f"{step.frame:5d}  {step.frame / fps:6.2f}s  {step.state.value:<10}"
               f"  cand={step.n_candidates}{speed}")
        cv2.putText(out, txt, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
        # legenda
        x = w - 215
        for label, col in (("medicao", BLUE), ("predicao", RED), ("estimativa", ORANGE)):
            cv2.circle(out, (x, 10), 4, col, -1)
            cv2.putText(out, label, (x + 7, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)
            x += 72
        if step.state == TrackState.DARK:
            cv2.rectangle(out, (0, h - 30), (w, h), RED, -1)
            cv2.putText(out, f"NO ESCURO ({step.dark} quadros sem medicao)", (10, h - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
