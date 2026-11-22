"""Filtro de Kalman implementado pela dupla (NumPy), com a notação da aula.

    estado      x  (n)            x = [px, py, vx, vy]^T  no modelo de velocidade constante
    transição   D  (n x n)        x_k = D x_{k-1} + ruído do processo  ~ N(0, Σd)
    medição     M  (m x n)        z_k = M x_k + ruído da medição       ~ N(0, Σm)

Predição:
    x⁻ = D x
    P⁻ = D P Dᵀ + Σd
Correção:
    y  = z − M x⁻                (inovação)
    S  = M P⁻ Mᵀ + Σm            (covariância da inovação)
    K  = P⁻ Mᵀ S⁻¹               (ganho de Kalman)
    x  = x⁻ + K y
    P  = (I − K M) P⁻

`cv2.KalmanFilter` é usado só no teste de validação (tests/test_kalman_vs_opencv.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class KalmanFilter:
    def __init__(self, D: np.ndarray, M: np.ndarray, Sd: np.ndarray, Sm: np.ndarray,
                 x0: np.ndarray, P0: np.ndarray):
        self.D = np.asarray(D, float)
        self.M = np.asarray(M, float)
        self.Sd = np.asarray(Sd, float)
        self.Sm = np.asarray(Sm, float)
        n, m = self.D.shape[0], self.M.shape[0]
        assert self.D.shape == (n, n) and self.M.shape == (m, n)
        assert self.Sd.shape == (n, n) and self.Sm.shape == (m, m)
        self.x = np.asarray(x0, float).reshape(n).copy()
        self.P = np.asarray(P0, float).reshape(n, n).copy()
        self._I = np.eye(n)
        # últimos valores, úteis para visualização e análise
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()

    # --------------------------------------------------------------------- passos
    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """x⁻ = D x ;  P⁻ = D P Dᵀ + Σd.  Sem correção depois, x⁻ vira a estimativa."""
        self.x = self.D @ self.x
        self.P = self.D @ self.P @ self.D.T + self.Sd
        self.x_pred, self.P_pred = self.x.copy(), self.P.copy()
        return self.x, self.P

    def innovation(self, z: np.ndarray, Sm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """y = z − M x⁻ ;  S = M P⁻ Mᵀ + Σm  (calculados sobre a predição atual)."""
        Sm = self.Sm if Sm is None else np.asarray(Sm, float)
        y = np.asarray(z, float).reshape(-1) - self.M @ self.x_pred
        S = self.M @ self.P_pred @ self.M.T + Sm
        return y, S

    def correct(self, z: np.ndarray, Sm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Correção com a medição z. `Sm` permite um Σm diferente por quadro (ex.: em metros)."""
        y, S = self.innovation(z, Sm)
        K = self.P_pred @ self.M.T @ np.linalg.inv(S)
        self.x = self.x_pred + K @ y
        self.P = (self._I - K @ self.M) @ self.P_pred
        return self.x, self.P

    # ------------------------------------------------------------------ utilidades
    def predicted_measurement(self, Sm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(M x⁻, S) — centro e covariância da região onde a próxima medição deve cair."""
        Sm = self.Sm if Sm is None else np.asarray(Sm, float)
        return self.M @ self.x_pred, self.M @ self.P_pred @ self.M.T + Sm


# ---------------------------------------------------------------------- modelos
@dataclass
class CVModel:
    """Modelo de velocidade constante 2D com aceleração aleatória (ruído branco)."""
    D: np.ndarray
    M: np.ndarray
    Sd: np.ndarray
    Sm: np.ndarray


def constant_velocity(dt: float, sigma_a: float, sigma_m: float) -> CVModel:
    """Matrizes do modelo de velocidade constante.

    dt       passo de tempo (1/fps)
    sigma_a  desvio-padrão da aceleração não modelada (unidade/s²): o "botão" entre
             filtro teimoso (pequeno, elefante) e nervoso (grande, peixinho-dourado)
    sigma_m  desvio-padrão do ruído da medição de posição (unidade)

    Σd vem do modelo de aceleração aleatória: a velocidade sofre um "empurrão" a·dt e a
    posição a·dt²/2, com a ~ N(0, σa²). Por eixo:
        σa² [[dt⁴/4, dt³/2],
             [dt³/2, dt²  ]]
    """
    D = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], float)
    M = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], float)
    q = sigma_a ** 2
    pp, pv, vv = dt ** 4 / 4 * q, dt ** 3 / 2 * q, dt ** 2 * q
    Sd = np.array([[pp, 0, pv, 0],
                   [0, pp, 0, pv],
                   [pv, 0, vv, 0],
                   [0, pv, 0, vv]], float)
    Sm = np.eye(2) * sigma_m ** 2
    return CVModel(D, M, Sd, Sm)


def init_state(z: np.ndarray, sigma_m: float, sigma_v0: float) -> tuple[np.ndarray, np.ndarray]:
    """Estado inicial a partir da primeira medição: velocidade zero com incerteza alta."""
    x0 = np.array([z[0], z[1], 0.0, 0.0])
    P0 = np.diag([sigma_m ** 2, sigma_m ** 2, sigma_v0 ** 2, sigma_v0 ** 2])
    return x0, P0
