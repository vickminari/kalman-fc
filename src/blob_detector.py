"""Detectores de blob alternativos, para comparar com o detector HSV (src/detector.py).

Os dois partem de um mapa contínuo de "brancura" (0..1) dentro do campo, em vez da
máscara binária:

    brancura = rampa(V; V_min − 20 .. V_min + 20) · rampa(S; S_max + 15 .. S_max − 15) · campo

com os mesmos limiares adaptativos do detector HSV (relativos à grama do quadro).

1. LoGBlobDetector — detecção de blobs no espaço de escala (Lindeberg): a resposta do
   Laplaciano da Gaussiana normalizado, −σ²·∇²G_σ * I, tem máximo no centro de um blob
   claro de raio r ≈ σ√2. Procuramos máximos locais em (x, y, σ). Vantagem esperada: acha
   a bola mesmo quando a máscara binária a funde com um pé ou a quebra em pedaços, e
   devolve a escala (tamanho) de graça.

2. OpenCVBlobDetector — cv2.SimpleBlobDetector: binariza em vários limiares, junta os
   contornos estáveis e filtra por área, circularidade, convexidade e inércia. É a versão
   "pronta" de blob do OpenCV (permitida: detectores prontos podem ser usados como sensor).

Os dois aplicam depois o mesmo teste do anel de grama do detector HSV, para a
comparação medir só o gerador de candidatos.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.detector import BallDetector, Candidate, Detection, DetectorParams


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 em lo, 1 em hi (lo > hi inverte o sentido)."""
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


@dataclass
class BlobParams:
    sigmas: tuple[float, ...] = (1.0, 1.4, 2.0, 2.8, 4.0, 5.6)   # px a 640 de largura
    log_threshold: float = 0.12        # resposta mínima do LoG normalizado
    ring_grass_min: float = 0.5
    whiteness_min: float = 0.35        # brancura média dentro do disco
    reference_width: int = 640


class _BlobBase:
    def __init__(self, params: BlobParams | None = None, det_params: DetectorParams | None = None):
        self.p = params or BlobParams()
        self.hsv_det = BallDetector(det_params or DetectorParams())

    def _maps(self, frame_bgr: np.ndarray):
        d = self.hsv_det
        h, w = frame_bgr.shape[:2]
        scale = w / d.p.reference_width
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        grass = d.grass_mask(hsv)
        fmask, top = d.field_mask(grass, scale)
        s_max, v_min = d.white_thresholds(hsv, grass)
        S = hsv[..., 1].astype(np.float32)
        V = hsv[..., 2].astype(np.float32)
        white = _ramp(V, v_min - 20, v_min + 20) * _ramp(S, s_max + 15, s_max - 15)
        white *= (fmask > 0)
        return white.astype(np.float32), grass, fmask, top, scale

    def _describe(self, x: float, y: float, r: float, white: np.ndarray,
                  grass: np.ndarray, score_shape: float) -> Candidate | None:
        """Brancura no disco e grama no anel; None se reprovar."""
        h, w = white.shape
        R = int(np.ceil(r * 1.6)) + 2
        x0, x1 = max(0, int(x) - R), min(w, int(x) + R + 1)
        y0, y1 = max(0, int(y) - R), min(h, int(y) + R + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - x, yy - y)
        disk = d <= max(1.0, r)
        ring = (d > r + 1) & (d <= r * 1.6 + 2)
        if not disk.any() or not ring.any():
            return None
        whiteness = float(white[y0:y1, x0:x1][disk].mean())
        ring_grass = float((grass[y0:y1, x0:x1][ring] > 0).mean())
        if whiteness < self.p.whiteness_min or ring_grass < self.p.ring_grass_min:
            return None
        area = float(np.pi * r * r)
        return Candidate(float(x), float(y), area, int(2 * r), int(2 * r),
                         float(np.clip(score_shape, 0, 1)), whiteness, ring_grass)


class LoGBlobDetector(_BlobBase):
    def detect(self, frame_bgr: np.ndarray, keep_masks: bool = False) -> Detection:
        white, grass, fmask, top, scale = self._maps(frame_bgr)
        sig = [s * scale for s in self.p.sigmas]
        stack = []
        for s in sig:
            g = cv2.GaussianBlur(white, (0, 0), s)
            stack.append(-(s ** 2) * cv2.Laplacian(g, cv2.CV_32F, ksize=3))
        L = np.stack(stack)                                   # (n_sigmas, h, w)
        # máximo local em 3x3 no espaço e entre escalas vizinhas
        k = np.ones((3, 3), np.uint8)
        spatial_max = np.stack([cv2.dilate(l, k) for l in L])
        scale_max = spatial_max.copy()
        scale_max[1:] = np.maximum(scale_max[1:], spatial_max[:-1])
        scale_max[:-1] = np.maximum(scale_max[:-1], spatial_max[1:])
        peaks = (L >= scale_max) & (L > self.p.log_threshold)
        si, ys, xs = np.nonzero(peaks)
        cands = []
        for s_i, y, x in zip(si, ys, xs):
            r = sig[s_i] * np.sqrt(2)
            c = self._describe(float(x), float(y), r, white, grass,
                               score_shape=min(1.0, L[s_i, y, x] / 0.5))
            if c is not None:
                cands.append(c)
        cands = _nms(cands)
        masks = {"white_cont": (white * 255).astype(np.uint8), "field": fmask} if keep_masks else {}
        return Detection(cands, masks, top)


class OpenCVBlobDetector(_BlobBase):
    def __init__(self, params: BlobParams | None = None, det_params: DetectorParams | None = None):
        super().__init__(params, det_params)
        bp = cv2.SimpleBlobDetector_Params()
        bp.filterByColor, bp.blobColor = True, 255
        bp.minThreshold, bp.maxThreshold, bp.thresholdStep = 60, 230, 20
        bp.minRepeatability = 2
        bp.filterByArea, bp.minArea, bp.maxArea = True, 6, 900
        bp.filterByCircularity, bp.minCircularity = True, 0.4
        bp.filterByConvexity, bp.minConvexity = True, 0.6
        bp.filterByInertia, bp.minInertiaRatio = True, 0.3
        self._bp = bp
        self._det = cv2.SimpleBlobDetector_create(bp)

    def detect(self, frame_bgr: np.ndarray, keep_masks: bool = False) -> Detection:
        white, grass, fmask, top, scale = self._maps(frame_bgr)
        img = (white * 255).astype(np.uint8)
        kps = self._det.detect(img)
        cands = []
        for kp in kps:
            r = max(1.0, kp.size / 2)
            c = self._describe(kp.pt[0], kp.pt[1], r, white, grass, score_shape=1.0)
            if c is not None:
                cands.append(c)
        masks = {"white_cont": img, "field": fmask} if keep_masks else {}
        return Detection(cands, masks, top)


def _nms(cands: list[Candidate]) -> list[Candidate]:
    """Remove candidatos sobrepostos (fica o de maior score)."""
    cands = sorted(cands, key=lambda c: -c.score)
    keep: list[Candidate] = []
    for c in cands:
        if all(np.hypot(c.x - k.x, c.y - k.y) > max(c.diameter, k.diameter) / 2 for k in keep):
            keep.append(c)
    return keep
