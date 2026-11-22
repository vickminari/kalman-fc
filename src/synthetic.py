"""Gerador de sequências sintéticas para testar o filtro sem vídeo.

Simula uma bola rolando no plano (px), com:
- velocidade aproximadamente constante entre "toques" (chutes/passes mudam a velocidade);
- atrito opcional;
- medições com ruído gaussiano (o "detector");
- oclusões (intervalos sem medição da bola);
- candidatos falsos: alguns aleatórios por quadro (ruído) e alguns fixos (folhas no gramado).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticConfig:
    n_frames: int = 300
    fps: float = 30.0
    area: tuple[float, float] = (640.0, 360.0)
    start: tuple[float, float] = (100.0, 200.0)
    speed_range: tuple[float, float] = (40.0, 250.0)   # px/s depois de cada toque
    kick_every: tuple[int, int] = (25, 70)             # quadros entre toques
    friction: float = 0.0                               # fração da velocidade perdida por segundo
    sigma_m: float = 1.5                                # ruído da medição (px)
    occlusions: list[tuple[int, int]] = field(default_factory=list)  # [início, fim) sem medição
    miss_prob: float = 0.0                              # falha aleatória do detector
    clutter_per_frame: float = 0.0                      # candidatos falsos aleatórios (média Poisson)
    static_clutter: list[tuple[float, float]] = field(default_factory=list)
    seed: int = 0


@dataclass
class SyntheticSequence:
    config: SyntheticConfig
    truth: np.ndarray                  # (T, 4) [px, py, vx, vy]
    measured: np.ndarray               # (T, 2) medição da bola (nan se não detectada)
    candidates: list[np.ndarray]       # por quadro: (N, 2), bola + falsos, embaralhados
    ball_index: list[int]              # índice da bola em candidates[t] (-1 se ausente)

    @property
    def dt(self) -> float:
        return 1.0 / self.config.fps


def generate(cfg: SyntheticConfig) -> SyntheticSequence:
    rng = np.random.default_rng(cfg.seed)
    dt = 1.0 / cfg.fps
    W, H = cfg.area
    truth = np.zeros((cfg.n_frames, 4))
    p = np.array(cfg.start, float)
    v = _random_velocity(rng, cfg.speed_range)
    next_kick = int(rng.integers(*cfg.kick_every))
    for t in range(cfg.n_frames):
        if t == next_kick:
            v = _random_velocity(rng, cfg.speed_range)
            next_kick = t + int(rng.integers(*cfg.kick_every))
        truth[t] = [p[0], p[1], v[0], v[1]]
        p = p + v * dt
        v = v * (1.0 - cfg.friction * dt)
        # quica nas bordas para a bola não sair da área
        for i, lim in enumerate((W, H)):
            if p[i] < 10 or p[i] > lim - 10:
                v[i] = -v[i]
                p[i] = np.clip(p[i], 10, lim - 10)

    occluded = np.zeros(cfg.n_frames, bool)
    for a, b in cfg.occlusions:
        occluded[a:b] = True
    missed = rng.random(cfg.n_frames) < cfg.miss_prob
    visible = ~occluded & ~missed

    measured = np.full((cfg.n_frames, 2), np.nan)
    noise = rng.normal(0.0, cfg.sigma_m, size=(cfg.n_frames, 2))
    measured[visible] = truth[visible, :2] + noise[visible]

    static = np.array(cfg.static_clutter, float).reshape(-1, 2)
    candidates, ball_index = [], []
    for t in range(cfg.n_frames):
        k = rng.poisson(cfg.clutter_per_frame) if cfg.clutter_per_frame > 0 else 0
        rand = rng.uniform([0, 0], [W, H], size=(k, 2))
        stat = static + rng.normal(0.0, 0.5, size=static.shape)
        pts = [rand, stat]
        if visible[t]:
            pts.append(measured[t][None, :])
        c = np.vstack(pts) if pts else np.zeros((0, 2))
        order = rng.permutation(len(c))
        c = c[order]
        idx = int(np.where(order == len(c) - 1)[0][0]) if visible[t] else -1
        candidates.append(c)
        ball_index.append(idx)
    return SyntheticSequence(cfg, truth, measured, candidates, ball_index)


def _random_velocity(rng: np.random.Generator, speed_range: tuple[float, float]) -> np.ndarray:
    speed = rng.uniform(*speed_range)
    ang = rng.uniform(0, 2 * np.pi)
    return speed * np.array([np.cos(ang), np.sin(ang)])
