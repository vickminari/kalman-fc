"""Descritor de auto-similaridade local (LSS), de Shechtman & Irani (CVPR 2007),
"Matching Local Self-Similarities across Images and Videos".

Para um ponto q:
1. compara-se o patch central (5x5) com todos os patches de uma região em volta de q,
   por SSD de cores (espaço CIE L*a*b*), o que dá a "superfície de distâncias" SSD_q(x, y);
2. a superfície vira correlação:  S_q(x, y) = exp(−SSD_q / max(var_ruído, var_auto(q))),
   em que var_auto(q) é a maior SSD entre o patch central e os vizinhos a 1 px (assim,
   patches com bordas fortes toleram mais variação);
3. S_q vai para uma grade log-polar (ângulos × raios), e cada posição guarda o MÁXIMO da
   correlação (o que dá tolerância a pequenas deformações);
4. o vetor é normalizado para [0, 1].

Adaptação para este projeto: o artigo calcula o descritor densamente (a cada 5 px), com
região de raio ~40 px, e casa "conjuntos" de descritores de um molde inteiro. Aqui a bola
tem de 3 a 30 px, então:
- o descritor é calculado SÓ nos candidatos do detector (custo baixo);
- a região é escalada pelo tamanho do candidato (raio = 1,6 × diâmetro, reamostrada
  para 41x41), o que torna o descritor aproximadamente invariante à escala;
- o "molde" é a média dos descritores de bolas verdadeiras de um trecho de treino, e a
  similaridade é a distância L1 até esse molde.
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE = 41          # lado da região normalizada
PATCH = 5          # lado do patch central
N_ANG = 12
N_RAD = 3
R_MIN, R_MAX = 1.5, 18.0


def _bins() -> np.ndarray:
    """Índice da posição log-polar de cada ponto da superfície de correlação (−1 = fora)."""
    n = SIZE - PATCH + 1                  # 37x37 posições de patch
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    dx, dy = xx - c, yy - c
    r = np.hypot(dx, dy)
    ang = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * N_ANG
    edges = np.geomspace(R_MIN, R_MAX, N_RAD + 1)
    rad = np.digitize(r, edges) - 1
    idx = np.where((rad >= 0) & (rad < N_RAD), rad * N_ANG + np.minimum(ang.astype(int), N_ANG - 1), -1)
    return idx


_BIN_IDX = _bins()
_BIN_MASKS = [(_BIN_IDX == b) for b in range(N_ANG * N_RAD)]


def descriptor(lab: np.ndarray, x: float, y: float, diameter: float,
               var_noise: float = 1500.0) -> np.ndarray | None:
    """Descritor LSS (N_ANG·N_RAD,) no ponto (x, y) da imagem L*a*b* (float32)."""
    R = max(6.0, 1.6 * diameter)
    h, w = lab.shape[:2]
    src = np.float32([[x - R, y - R], [x + R, y - R], [x - R, y + R]])
    dst = np.float32([[0, 0], [SIZE - 1, 0], [0, SIZE - 1]])
    A = cv2.getAffineTransform(src, dst)
    region = cv2.warpAffine(lab, A, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    c0 = (SIZE - PATCH) // 2
    patch = region[c0:c0 + PATCH, c0:c0 + PATCH]
    ssd = cv2.matchTemplate(region, patch, cv2.TM_SQDIFF)      # (37, 37)
    m = ssd.shape[0] // 2
    var_auto = float(np.max(ssd[m - 1:m + 2, m - 1:m + 2]))
    S = np.exp(-ssd / max(var_noise, var_auto))
    d = np.array([S[mask].max() if mask.any() else 0.0 for mask in _BIN_MASKS], np.float32)
    lo, hi = d.min(), d.max()
    if hi - lo < 1e-6:
        return None
    return (d - lo) / (hi - lo)


def to_lab(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


class LSSVerifier:
    """Distância L1 entre o descritor de um candidato e o molde (média de bolas reais)."""

    def __init__(self, template: np.ndarray | None = None):
        self.template = template

    def fit(self, descriptors: np.ndarray) -> "LSSVerifier":
        self.template = np.nanmean(descriptors, axis=0)
        return self

    def distance(self, d: np.ndarray | None) -> float:
        if d is None or self.template is None:
            return float("nan")
        return float(np.abs(d - self.template).mean())
