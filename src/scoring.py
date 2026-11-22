"""Score combinado dos candidatos (o "quanto isto parece a bola").

Combina, numa regressão logística (classificador linear), atributos que o experimento
experiments/candidate_features.py mostrou que separam a bola dos falsos positivos:

    score, circularidade, brancura, grama no anel   (do detector HSV)
    movimento com a câmera compensada               (src/motion_cue.py)
    similaridade de auto-similaridade local (LSS)    (src/lss.py, Shechtman & Irani 2007)
    ajuste ao tamanho esperado pela linha da imagem   (prior de perspectiva: log d = a + b·y)

    z = w0 + Σ wi · (xi − μi) / σi        p = 1 / (1 + e^(−z))

Os pesos, a padronização, o molde LSS e o prior de tamanho são ajustados num trecho de
TREINO do vídeo e salvos em JSON (data/models/). No pipeline:
- candidatos com z abaixo do limiar (que mantinha 95% das bolas do treino) são descartados;
- p vira o score usado pela trilha (inicialização e desempate na associação).

Limitação: o prior de tamanho e o molde LSS dependem da posição da câmera. Para outro
vídeo, é preciso reajustar (ou, no caso do tamanho, derivá-lo da homografia do campo).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.detector import Candidate
from src.lss import descriptor


@dataclass
class BallScorer:
    features: list[str]
    mean: list[float]
    std: list[float]
    weights: list[float]          # [bias, w1, ..., wn]
    fill: list[float]             # valor para atributo ausente (mediana do treino)
    threshold: float              # z mínimo para manter o candidato
    size_prior: list[float]       # [a, b] em log(d) = a + b·y
    lss_template: list[float]
    reference_width: int = 640
    trained_on: str = ""

    # ---------------------------------------------------------------- persistência
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BallScorer":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    # ---------------------------------------------------------------- atributos
    def feature_matrix(self, cands: list[Candidate], motion: np.ndarray,
                       lab: np.ndarray | None, width: int) -> np.ndarray:
        """Atributos (N, n_features), na ordem de `self.features`."""
        s = width / self.reference_width        # o prior e o molde valem a 640 px
        a, b = self.size_prior
        tmpl = np.asarray(self.lss_template)
        cols = []
        for f in self.features:
            if f == "score":
                v = [c.score for c in cands]
            elif f in ("circularity", "whiteness", "ring_grass", "cool"):
                v = [getattr(c, f) for c in cands]
            elif f == "motion":
                v = np.log1p(motion)
            elif f == "size_fit":
                v = [-abs(np.log(c.diameter / s) - (a + b * c.y / s)) for c in cands]
            elif f == "lss_sim":
                v = []
                for c in cands:
                    d = descriptor(lab, c.x, c.y, c.diameter) if lab is not None else None
                    v.append(np.nan if d is None else -float(np.abs(d - tmpl).mean()))
            else:
                raise ValueError(f"atributo desconhecido: {f}")
            cols.append(np.asarray(v, float))
        X = np.stack(cols, axis=1) if cols else np.zeros((len(cands), 0))
        fill = np.asarray(self.fill)
        nan = np.isnan(X)
        X[nan] = np.broadcast_to(fill, X.shape)[nan]
        return X

    def z(self, X: np.ndarray) -> np.ndarray:
        Z = (X - np.asarray(self.mean)) / np.asarray(self.std)
        w = np.asarray(self.weights)
        return w[0] + Z @ w[1:]

    @staticmethod
    def prob(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))
