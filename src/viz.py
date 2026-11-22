"""Sobreposições do vídeo de saída, com as cores da aula:

    medição               azul
    predição              vermelho  (+ elipse da região de busca)
    estimativa corrigida  laranja   (+ rastro da trajetória)
    "NO ESCURO"           faixa vermelha quando não há medição aceita

Visualização honesta da predição (report 012): o rastro laranja SÓLIDO liga apenas
quadros com medição aceita. Quando não há medição, a estimativa é só a predição do
filtro (x⁻ = D x): ela aparece TRACEJADA e esmaecida, a partir do último ponto medido, e
deixa de ser desenhada depois de `max_drawn_dark` quadros (a trilha continua viva até
`max_dark`, e a região de busca continua aparecendo). Assim a predição andando com uma
velocidade mal estimada não se confunde com a trajetória da bola.

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


# Um item do rastro: (ponto, com_medição) ou None (separa trilhas diferentes).
TrailItem = tuple[np.ndarray, bool] | None


def trail_segments(trail, max_drawn_dark: int, keep_past_dark: bool = False
                   ) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Separa o rastro em segmentos sólidos (medição) e tracejados (predição no escuro).

    - sólido: quadros seguidos COM medição aceita, na mesma trilha;
    - tracejado: quadros seguidos SEM medição, começando no último ponto medido (se houver)
      e cortados depois de `max_drawn_dark` quadros. Uma retomada não é ligada ao trecho
      anterior: o salto da predição até a medição não vira linha.
    Por padrão só o trecho no escuro EM ANDAMENTO (no fim do rastro) é devolvido: quando a
    bola volta ou a trilha morre, a predição daquele trecho some, e o histórico fica só
    com o que foi medido. `keep_past_dark=True` mantém os trechos antigos.
    Devolve (sólidos, tracejados); cada segmento é uma lista com 2 pontos ou mais.
    """
    solid: list[list[np.ndarray]] = []
    dashed: list[list[np.ndarray]] = []
    seg: list[np.ndarray] = []           # segmento sólido em construção
    dseg: list[np.ndarray] = []          # segmento tracejado em construção
    last_meas: np.ndarray | None = None  # último ponto medido da trilha atual
    n_dark = 0

    def flush(at_end: bool = False):
        nonlocal seg, dseg
        if len(seg) >= 2:
            solid.append(seg)
        if len(dseg) >= 2 and (at_end or keep_past_dark):
            dashed.append(dseg)
        seg, dseg = [], []

    for item in trail:
        if item is None:                 # trilha nova (ou sem trilha): corta tudo
            flush()
            last_meas, n_dark = None, 0
            continue
        p, measured = item
        if measured:
            if dseg:                     # fim de um trecho no escuro: não liga a retomada
                flush()
            seg.append(p)
            last_meas, n_dark = p, 0
        else:
            if seg:
                flush()
            n_dark += 1
            if n_dark > max_drawn_dark:
                continue
            if not dseg and last_meas is not None:
                dseg.append(last_meas)
            dseg.append(p)
    flush(at_end=True)
    return solid, dashed


def dashed_polyline(img: np.ndarray, pts: np.ndarray, color, thickness: int = 1,
                    dash: float = 5.0, gap: float = 4.0) -> None:
    """Polilinha tracejada (o OpenCV não tem): percorre os segmentos alternando traço e vão."""
    pts = np.asarray(pts, float)
    on, left = True, dash
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.linalg.norm(b - a))
        pos = 0.0
        while pos < L:
            step = min(left, L - pos)
            if on:
                p0 = a + (b - a) * (pos / L)
                p1 = a + (b - a) * ((pos + step) / L)
                cv2.line(img, _pt(p0), _pt(p1), color, thickness, cv2.LINE_AA)
            pos += step
            left -= step
            if left <= 1e-9:
                on = not on
                left = dash if on else gap


def blend(dst: np.ndarray, layer: np.ndarray, alpha: float) -> None:
    """Mistura `layer` sobre `dst` com opacidade `alpha`, só onde `layer` tem desenho
    (esmaecido: o fundo continua aparecendo por baixo)."""
    m = layer.any(axis=2)
    dst[m] = (alpha * layer[m] + (1 - alpha) * dst[m]).astype(dst.dtype)


class Overlay:
    def __init__(self, trail: int = 45, gate_threshold: float = 9.21, max_drawn_dark: int = 10,
                 dark_alpha: float = 0.45):
        self.trail: deque[TrailItem] = deque(maxlen=trail)
        self.gate_threshold = gate_threshold
        self.max_drawn_dark = max_drawn_dark   # quadros no escuro em que a predição ainda é desenhada
        self.dark_alpha = dark_alpha           # opacidade da predição no escuro

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
        measured = step.z is not None
        if step.new_track:
            self.trail.append(None)
        self.trail.append(None if step.x is None else (step.x[:2].copy(), measured))
        solid, dashed = trail_segments(self.trail, self.max_drawn_dark)
        for seg in solid:
            cv2.polylines(out, [to_cur(np.array(seg)).astype(np.int32)], False, ORANGE, 1, cv2.LINE_AA)

        # predição no escuro: numa camada à parte, misturada com transparência (esmaecida)
        show_pred = step.x is not None and not measured and step.dark <= self.max_drawn_dark
        if dashed or show_pred:
            layer = np.zeros_like(out)
            for seg in dashed:
                dashed_polyline(layer, to_cur(np.array(seg)), ORANGE, 1)
            if show_pred:
                cv2.circle(layer, _pt(to_cur(step.x[:2])[0]), 4, ORANGE, 1, cv2.LINE_AA)
            blend(out, layer, self.dark_alpha)

        # região de busca: sempre (é onde o filtro procura); o centro previsto só enquanto
        # a predição ainda é desenhada
        if step.z_pred is not None and step.S is not None:
            ell = to_cur(ellipse(step.z_pred, step.S, self.gate_threshold)).astype(np.int32)
            cv2.polylines(out, [ell], True, RED, 1, cv2.LINE_AA)
            if measured or step.dark <= self.max_drawn_dark:
                cv2.drawMarker(out, _pt(to_cur(step.z_pred)[0]), RED, cv2.MARKER_CROSS, 10, 2)

        if step.z_index is not None and len(candidates_cur):
            cv2.circle(out, _pt(candidates_cur[step.z_index]), 7, BLUE, 2, cv2.LINE_AA)

        # estimativa corrigida (laranja cheio) só quando houve correção com medição
        if step.x is not None and measured:
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
            extra = "" if step.dark <= self.max_drawn_dark else " - predicao oculta"
            cv2.putText(out, f"NO ESCURO ({step.dark} quadros sem medicao{extra})", (10, h - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
