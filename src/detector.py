"""Detector da bola (o "sensor" do filtro de Kalman).

Versão 0 — só aparência, sem subtração de fundo. A bola é branca com detalhes
coloridos e quase sempre rasteira, então ela aparece como uma mancha branca, compacta
e cercada de grama:

1. máscara da grama no HSV;
2. região do campo: para cada coluna, a primeira linha (de cima para baixo) em que a
   grama domina; uma abertura morfológica 1D tira os jogadores que tampam a borda;
3. máscara "branca" (saturação baixa, valor alto) dentro do campo;
4. fechamento (preenche os detalhes coloridos da bola) e abertura (tira ruído);
5. componentes conectados filtrados por área, circularidade e razão de aspecto;
6. fração de grama no anel em volta da mancha (descarta rodapé, rede do gol e
   manchas grudadas em jogadores).

Limitações conhecidas: pés e canelas claros, folhas/papéis brancos no gramado e a base
da rede do gol também viram candidatos. Quem escolhe a bola entre eles é o gating
do filtro de Kalman.

A subtração de fundo (MOG2) fica desligada por padrão porque os vídeos de teste foram
gravados com a câmera se movendo; ela só faz sentido com a câmera fixa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DetectorParams:
    # grama (OpenCV: H em 0..180)
    grass_h: tuple[int, int] = (25, 60)
    grass_s_min: int = 50
    grass_v_min: int = 30
    # borda do campo
    field_density: float = 0.6       # fração de grama na janela vertical para considerar "campo"
    field_window: int = 9            # altura da janela vertical (px na resolução de referência)
    field_smooth: int = 121          # janela da abertura 1D que tira jogadores da borda (colunas)
    field_margin: int = 0            # px acima da borda que ainda contam como campo
    # pixels "brancos" da bola — limiares adaptativos, relativos à grama do quadro:
    #   S < min(white_s_max, white_s_ratio · S_mediana_grama)
    #   V > max(white_v_min, V_mediana_grama + white_v_offset)
    # (a exposição muda entre vídeos e ao longo do vídeo; grama clara não pode virar "branco")
    white_s_max: int = 70
    white_s_ratio: float = 0.5
    white_v_min: int = 120
    white_v_offset: int = 30
    # forma (área em px² na resolução de referência de 640 px de largura)
    area_min: float = 8.0
    area_max: float = 900.0
    circularity_min: float = 0.40
    aspect_max: float = 2.0
    close_size: int = 5              # fechamento que preenche os detalhes coloridos da bola
    # aparência
    whiteness_min: float = 0.5       # fração de pixels brancos dentro da mancha
    ring_grass_min: float = 0.50     # fração de grama no anel em volta da mancha
    ring_width: float = 0.6          # largura do anel, em raios da mancha (mín. 2 px)
    reference_width: int = 640
    # cor da bola (report 015): fração de pixels de matiz "fria" no disco do candidato.
    # Sob os refletores o branco da bola puxa para o azul e a estampa é azul; pele, calção,
    # parede e grama ficam em matizes quentes. Não filtra: vira atributo do score combinado.
    cool_h: tuple[int, int] = (85, 150)
    cool_v_min: int = 60
    # subtração de fundo (só com câmera fixa)
    use_mog2: bool = False


@dataclass
class Candidate:
    x: float
    y: float
    area: float
    w: int
    h: int
    circularity: float
    whiteness: float
    ring_grass: float
    cool: float = 0.0            # fração de matiz fria no disco (cor da bola; report 015)

    @property
    def score(self) -> float:
        return self.circularity * self.whiteness * self.ring_grass

    @property
    def diameter(self) -> float:
        return 2.0 * np.sqrt(self.area / np.pi)


@dataclass
class Detection:
    candidates: list[Candidate]
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    field_top: np.ndarray | None = None  # linha da borda do campo por coluna

    @property
    def best(self) -> Candidate | None:
        return max(self.candidates, key=lambda c: c.score) if self.candidates else None


class BallDetector:
    def __init__(self, params: DetectorParams | None = None):
        self.p = params or DetectorParams()
        self._mog2 = None
        if self.p.use_mog2:
            self._mog2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25,
                                                            detectShadows=True)

    # ------------------------------------------------------------------ máscaras
    def grass_mask(self, hsv: np.ndarray) -> np.ndarray:
        p = self.p
        lo = np.array([p.grass_h[0], p.grass_s_min, p.grass_v_min], np.uint8)
        hi = np.array([p.grass_h[1], 255, 255], np.uint8)
        return cv2.inRange(hsv, lo, hi)

    def field_mask(self, grass: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
        """Região do campo a partir da borda superior da grama em cada coluna."""
        p = self.p
        h, w = grass.shape
        win = max(3, int(round(p.field_window * scale)))
        dens = cv2.blur((grass > 0).astype(np.float32), (1, win))
        is_field = dens >= p.field_density
        has = is_field.any(axis=0)
        top = np.where(has, is_field.argmax(axis=0), h).astype(np.float32)
        # jogadores na frente da borda "empurram" a borda para baixo (picos em y);
        # a abertura morfológica 1D (mín. seguido de máx.) remove picos mais estreitos que
        # a janela e preserva bordas inclinadas.
        k = max(3, int(round(p.field_smooth * scale)) | 1)
        top = _median_1d(top, 5)
        top = _max_1d(_min_1d(top, k), k)
        top = np.clip(top - p.field_margin * scale, 0, h)
        rows = np.arange(h, dtype=np.float32)[:, None]
        mask = (rows >= top[None, :]).astype(np.uint8) * 255
        return mask, top

    def white_thresholds(self, hsv: np.ndarray, grass: np.ndarray) -> tuple[int, int]:
        """(S máx., V mín.) dos pixels brancos, relativos à grama deste quadro."""
        p = self.p
        g = hsv[::4, ::4][grass[::4, ::4] > 0]  # amostra esparsa basta para a mediana
        if len(g) < 100:
            return p.white_s_max, p.white_v_min
        s_med, v_med = np.median(g[:, 1]), np.median(g[:, 2])
        s_max = int(min(p.white_s_max, p.white_s_ratio * s_med))
        v_min = int(min(250, max(p.white_v_min, v_med + p.white_v_offset)))
        return s_max, v_min

    def white_mask(self, hsv: np.ndarray, grass: np.ndarray | None = None) -> np.ndarray:
        p = self.p
        s_max, v_min = (p.white_s_max, p.white_v_min) if grass is None \
            else self.white_thresholds(hsv, grass)
        lo = np.array([0, 0, v_min], np.uint8)
        hi = np.array([180, s_max, 255], np.uint8)
        return cv2.inRange(hsv, lo, hi)

    # ------------------------------------------------------------------ detecção
    def detect(self, frame_bgr: np.ndarray, keep_masks: bool = False) -> Detection:
        p = self.p
        h, w = frame_bgr.shape[:2]
        scale = w / p.reference_width
        area_scale = scale * scale

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        grass = self.grass_mask(hsv)
        fmask, top = self.field_mask(grass, scale)
        white = self.white_mask(hsv, grass)
        cand = cv2.bitwise_and(white, fmask)

        if self._mog2 is not None:
            fg = self._mog2.apply(frame_bgr)
            fg = np.where(fg == 255, 255, 0).astype(np.uint8)  # descarta sombras (127)
            cand = cv2.bitwise_and(cand, fg)

        # fechamento preenche os detalhes coloridos da bola; abertura tira ruído da grama
        kc = max(3, int(round(p.close_size * scale)) | 1)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
        cands: list[Candidate] = []
        for i in range(1, n):
            x0, y0, bw, bh, area = stats[i]
            if not (p.area_min * area_scale <= area <= p.area_max * area_scale):
                continue
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > p.aspect_max:
                continue
            r = max(bw, bh) / 2.0
            circ = float(area / (np.pi * r * r)) if r > 0 else 0.0
            if circ < p.circularity_min:
                continue

            blob = labels[y0:y0 + bh, x0:x0 + bw] == i
            whiteness = float(np.count_nonzero(white[y0:y0 + bh, x0:x0 + bw][blob])) / float(area)
            if whiteness < p.whiteness_min:
                continue

            # anel em volta da mancha: precisa ser majoritariamente grama
            m = max(2, int(round(p.ring_width * r)))
            ya, yb = max(0, y0 - m), min(h, y0 + bh + m)
            xa, xb = max(0, x0 - m), min(w, x0 + bw + m)
            ring = labels[ya:yb, xa:xb] != i
            ring_grass = float(np.count_nonzero(grass[ya:yb, xa:xb][ring])) / max(1, np.count_nonzero(ring))
            if ring_grass < p.ring_grass_min:
                continue

            cx, cy = cents[i]
            cool = cool_fraction(hsv, cx, cy, np.sqrt(area / np.pi), p.cool_h, p.cool_v_min)
            cands.append(Candidate(float(cx), float(cy), float(area), int(bw), int(bh),
                                   min(circ, 1.0), whiteness, ring_grass, cool))

        masks = {}
        if keep_masks:
            masks = {"grass": grass, "field": fmask, "white": white, "candidates": cand}
        return Detection(cands, masks, top)


def cool_fraction(hsv: np.ndarray, x: float, y: float, r: float,
                  h_range: tuple[int, int] = (85, 150), v_min: int = 60) -> float:
    """Fração dos pixels do disco (centro x, y; raio r, mín. 2 px) com matiz entre
    h_range (ciano/azul/violeta no OpenCV, H em 0..180) e brilho ≥ v_min, com qualquer
    saturação. É o atributo `cool` do candidato (report 015)."""
    r = max(float(r), 2.0)
    R = int(np.ceil(r)) + 1
    xi, yi = int(round(x)), int(round(y))
    H, W = hsv.shape[:2]
    x0, x1, y0, y1 = max(xi - R, 0), min(xi + R + 1, W), max(yi - R, 0), min(yi + R + 1, H)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = np.hypot(xx - x, yy - y) <= r
    if not disc.any():
        return 0.0
    crop = hsv[y0:y1, x0:x1][disc]
    ok = (crop[:, 0] >= h_range[0]) & (crop[:, 0] <= h_range[1]) & (crop[:, 2] >= v_min)
    return float(ok.mean())


def _window(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    return np.lib.stride_tricks.sliding_window_view(np.pad(x, pad, mode="edge"), k)


def _median_1d(x: np.ndarray, k: int) -> np.ndarray:
    return np.median(_window(x, k), axis=1).astype(np.float32)


def _min_1d(x: np.ndarray, k: int) -> np.ndarray:
    return _window(x, k).min(axis=1).astype(np.float32)


def _max_1d(x: np.ndarray, k: int) -> np.ndarray:
    return _window(x, k).max(axis=1).astype(np.float32)


def draw_candidates(img: np.ndarray, det: Detection, color_best=(255, 0, 0),
                    color_other=(200, 200, 200), field_color=(0, 255, 255)) -> np.ndarray:
    """Sobrepõe borda do campo e candidatos (melhor em azul, como a medição na aula)."""
    out = img.copy()
    if det.field_top is not None:
        pts = np.stack([np.arange(len(det.field_top)), det.field_top], axis=1).astype(np.int32)
        cv2.polylines(out, [pts], False, field_color, 1)
    best = det.best
    for c in det.candidates:
        col = color_best if c is best else color_other
        r = max(4, int(c.diameter / 2) + 3)
        cv2.circle(out, (int(round(c.x)), int(round(c.y))), r, col, 1 if c is not best else 2)
    return out
