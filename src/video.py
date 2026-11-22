"""Leitura de vídeo (arquivo ou câmera) com redimensionamento para a resolução de processamento."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def open_video(source: str | int) -> cv2.VideoCapture:
    """Abre um arquivo de vídeo ou uma câmera (índice inteiro)."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, str) and not Path(source).exists():
        raise FileNotFoundError(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"não foi possível abrir {source!r}")
    return cap


def video_info(cap: cv2.VideoCapture, path: str = "") -> VideoInfo:
    return VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def processing_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    """Tamanho de processamento mantendo a proporção (ex.: 848x480 -> 640x362)."""
    if target_width <= 0 or target_width >= width:
        return width, height
    scale = target_width / width
    return target_width, int(round(height * scale))


def frames(cap: cv2.VideoCapture, target_width: int = 640,
           start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Gera (índice, quadro redimensionado) a partir de `start`."""
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = processing_size(w, h, target_width)
    idx = start
    while stop is None or idx < stop:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % step == 0:
            if size != (w, h):
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            yield idx, frame
        idx += 1
