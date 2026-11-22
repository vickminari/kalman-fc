"""Região de busca (gating) derivada da covariância prevista.

A medição esperada é ẑ = M x⁻, com covariância da inovação S = M P⁻ Mᵀ + Σm.
Um candidato z é aceito se a distância de Mahalanobis ao quadrado

    d² = (z − ẑ)ᵀ S⁻¹ (z − ẑ)

ficar abaixo de um limiar χ² com 2 graus de liberdade (a medição tem 2 componentes).
Se a medição realmente vem da bola, d² segue χ²(2), então o limiar de 5,99 aceita a bola
verdadeira 95% das vezes, e o de 9,21 em 99% das vezes. Geometricamente, a região aceita é
uma elipse centrada em ẑ, que cresce quando o filtro fica "no escuro" (P⁻ aumenta).
"""
from __future__ import annotations

import numpy as np

CHI2_2DOF = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210, 0.999: 13.816}


def chi2_threshold(prob: float) -> float:
    """Limiar χ² (2 g.l.) para a probabilidade de aceitar a bola verdadeira."""
    if prob in CHI2_2DOF:
        return CHI2_2DOF[prob]
    return float(-2.0 * np.log(1.0 - prob))   # forma fechada para 2 graus de liberdade


def mahalanobis2(z: np.ndarray, z_pred: np.ndarray, S: np.ndarray) -> np.ndarray:
    """d² de cada candidato (N x 2) até a medição prevista."""
    z = np.atleast_2d(np.asarray(z, float))
    y = z - np.asarray(z_pred, float).reshape(1, 2)
    Si = np.linalg.inv(S)
    return np.einsum("ni,ij,nj->n", y, Si, y)


def select(candidates: np.ndarray, z_pred: np.ndarray, S: np.ndarray,
           threshold: float) -> tuple[int | None, np.ndarray]:
    """Vizinho mais próximo (em Mahalanobis) dentro da região de busca.

    Retorna (índice do candidato aceito ou None, d² de todos os candidatos).
    """
    candidates = np.asarray(candidates, float).reshape(-1, 2)
    if len(candidates) == 0:
        return None, np.zeros(0)
    d2 = mahalanobis2(candidates, z_pred, S)
    i = int(np.argmin(d2))
    return (i if d2[i] <= threshold else None), d2


def ellipse(z_pred: np.ndarray, S: np.ndarray, threshold: float,
            n: int = 48) -> np.ndarray:
    """Pontos (n x 2) do contorno da região de busca {z : d²(z) = limiar}."""
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, 1e-12)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    pts = circle * np.sqrt(vals * threshold)        # semi-eixos = sqrt(λ · limiar)
    return pts @ vecs.T + np.asarray(z_pred, float).reshape(1, 2)
