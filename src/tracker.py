"""Lógica de trilha da bola: inicialização, gating, "no escuro", perda e reinício.

Estados:
    BUSCANDO    sem trilha; o próximo candidato plausível inicializa o filtro
    TENTATIVA   trilha nova, ainda não confirmada (precisa de `confirm_hits` medições aceitas)
    RASTREANDO  trilha confirmada com medição aceita neste quadro
    NO_ESCURO   trilha confirmada sem medição aceita: o filtro só prediz e a região de
                busca cresce sozinha (P⁻ aumenta a cada predição)
Depois de `max_dark` quadros no escuro, a trilha é perdida e o estado volta a BUSCANDO.

As coordenadas são as que o chamador fornecer. No pipeline, são pixels do quadro de
referência (estabilizado), então o movimento da câmera não vira velocidade da bola.

Associação: entre os candidatos dentro da região de busca (d² ≤ limiar χ²), fica o de
menor custo d² − w·ln(score). A distância de Mahalanobis manda, e a aparência (score
do detector) desempata entre a bola e um pé ou uma folha próximos.

Mapa de desordem estática: folhas, papéis e a base da rede aparecem sempre no mesmo
lugar (nas coordenadas estabilizadas). Uma média móvel da ocupação de cada célula, sem
contar a medição aceita pela trilha (para a bola parada não virar "desordem"), marca
esses pontos. Eles perdem prioridade na inicialização. Uma trilha parada há mais de
`static_max_s` segundos, com outro candidato disponível, é descartada (provavelmente
está presa numa folha) e o lugar é marcado como desordem.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.gating import chi2_threshold, mahalanobis2
from src.kalman import KalmanFilter, constant_velocity, init_state


class TrackState(str, Enum):
    SEARCHING = "BUSCANDO"
    TENTATIVE = "TENTATIVA"
    TRACKING = "RASTREANDO"
    DARK = "NO_ESCURO"


@dataclass
class TrackerParams:
    fps: float = 30.0
    sigma_a: float = 4000.0      # px/s² — aceleração não modelada (chutes são bruscos)
    sigma_m: float = 1.5         # px — ruído da medição (medir no V2)
    sigma_v0: float = 300.0      # px/s — incerteza da velocidade inicial
    gate_prob: float = 0.99      # probabilidade de aceitar a bola verdadeira no gating
    max_dark: int = 20           # quadros só com predição antes de perder a trilha
    confirm_hits: int = 3        # medições aceitas para confirmar uma trilha nova
    tentative_max_dark: int = 2  # trilha tentativa morre se ficar este tempo sem medição
    max_gate_px: float = 60.0    # teto do raio da região de busca (evita "engolir" um pé)
    score_weight: float = 2.0    # peso da aparência no custo de associação
    # salvaguardas físicas (fora do filtro de Kalman propriamente dito)
    max_speed: float = 1e9       # px/s — velocidade máxima plausível (1e9 = desligado)
    dark_damping: float = 1.0    # no escuro, multiplica a velocidade a cada quadro (1 = desligado)
    # mapa de desordem estática
    clutter_cell: float = 6.0    # px
    clutter_tau_s: float = 3.0   # constante de tempo da média móvel
    clutter_level: float = 0.5   # ocupação acima disso = desordem estática
    static_radius: float = 4.0   # px — trilha que não sai deste raio é considerada parada
    static_max_s: float = 4.0    # parada há mais tempo que isso, com alternativas -> descarta


@dataclass
class TrackStep:
    frame: int
    state: TrackState
    n_candidates: int
    z: np.ndarray | None            # medição aceita (None se nenhuma)
    z_index: int | None             # índice do candidato aceito
    x_pred: np.ndarray | None       # estado previsto x⁻
    z_pred: np.ndarray | None       # medição prevista M x⁻ (centro da região de busca)
    S: np.ndarray | None            # covariância da inovação (forma da região de busca)
    x: np.ndarray | None            # estimativa depois da correção (ou só predição)
    P: np.ndarray | None
    d2: np.ndarray | None           # Mahalanobis² de todos os candidatos
    dark: int                       # quadros seguidos sem medição
    new_track: bool = False
    lost: bool = False
    lost_reason: str = ""


class ClutterMap:
    """Média móvel da ocupação de células por candidatos (coordenadas estabilizadas)."""

    def __init__(self, cell: float, alpha: float):
        self.cell = cell
        self.alpha = alpha
        self.occ: dict[tuple[int, int], float] = {}

    def _key(self, p) -> tuple[int, int]:
        return int(np.floor(p[0] / self.cell)), int(np.floor(p[1] / self.cell))

    def update(self, pts: np.ndarray) -> None:
        a = self.alpha
        hit = {self._key(p) for p in pts}
        for k in list(self.occ):
            self.occ[k] *= (1 - a)
            if self.occ[k] < 1e-3 and k not in hit:
                del self.occ[k]
        for k in hit:
            self.occ[k] = self.occ.get(k, 0.0) + a

    def mark(self, p) -> None:
        """Marca um ponto como desordem estática (ocupação máxima)."""
        self.occ[self._key(p)] = 1.0

    def level(self, p) -> float:
        """Ocupação somada na vizinhança 3x3, limitada a 1. Um objeto parado que cai na
        borda de células divide a ocupação entre elas, então a soma é o que conta."""
        i, j = self._key(p)
        return min(1.0, sum(self.occ.get((i + di, j + dj), 0.0)
                            for di in (-1, 0, 1) for dj in (-1, 0, 1)))


class BallTracker:
    def __init__(self, params: TrackerParams | None = None):
        self.p = params or TrackerParams()
        self.model = constant_velocity(1.0 / self.p.fps, self.p.sigma_a, self.p.sigma_m)
        self.threshold = chi2_threshold(self.p.gate_prob)
        self.kf: KalmanFilter | None = None
        self.state = TrackState.SEARCHING
        self.hits = 0
        self.dark = 0
        self.still = 0               # quadros seguidos com a trilha parada
        self.anchor: np.ndarray | None = None  # onde a trilha parou
        self.frame = -1
        self.clutter = ClutterMap(self.p.clutter_cell, 1.0 / (self.p.clutter_tau_s * self.p.fps))

    # ------------------------------------------------------------------ util
    def _gate_S(self, S: np.ndarray) -> np.ndarray:
        """Limita a região de busca a `max_gate_px` de raio (no eixo maior)."""
        lam = np.linalg.eigvalsh(S).max()
        r = np.sqrt(lam * self.threshold)
        if r <= self.p.max_gate_px:
            return S
        return S * (self.p.max_gate_px / r) ** 2

    def _score(self, scores: np.ndarray | None, n: int) -> np.ndarray:
        if scores is None or len(scores) != n:
            return np.ones(n)
        return np.clip(np.asarray(scores, float), 1e-3, 1.0)

    def _start(self, cands: np.ndarray, scores: np.ndarray) -> int | None:
        """Escolhe o candidato que inicializa a trilha: maior score, fora da desordem estática."""
        if len(cands) == 0:
            return None
        s = scores.copy()
        static = np.array([self.clutter.level(c) > self.p.clutter_level for c in cands])
        if (~static).any():
            s[static] = -np.inf
        return int(np.argmax(s))

    def _associate(self, cands: np.ndarray, scores: np.ndarray, z_pred: np.ndarray,
                   S: np.ndarray) -> tuple[int | None, np.ndarray]:
        if len(cands) == 0:
            return None, np.zeros(0)
        d2 = mahalanobis2(cands, z_pred, S)
        inside = d2 <= self.threshold
        if not inside.any():
            return None, d2
        cost = d2 - self.p.score_weight * np.log(scores)
        cost[~inside] = np.inf
        return int(np.argmin(cost)), d2

    def _drop(self) -> None:
        self.kf = None
        self.state = TrackState.SEARCHING
        self.hits = 0
        self.dark = 0
        self.still = 0
        self.anchor = None

    # ------------------------------------------------------------------ passo
    def step(self, candidates: np.ndarray, scores: np.ndarray | None = None,
             Sm: np.ndarray | None = None) -> TrackStep:
        """Processa um quadro. `candidates`: (N, 2) posições medidas; `scores`: (N,) opcional,
        em (0, 1]; `Sm`: Σm deste quadro (se diferente do padrão, ex.: propagado por homografia)."""
        self.frame += 1
        cands = np.asarray(candidates, float).reshape(-1, 2)
        sc = self._score(scores, len(cands))

        # ---- sem trilha: tenta inicializar
        if self.kf is None:
            i = self._start(cands, sc)
            self.clutter.update(cands if i is None else np.delete(cands, i, axis=0))
            if i is None:
                return TrackStep(self.frame, self.state, len(cands), None, None, None, None,
                                 None, None, None, None, 0)
            x0, P0 = init_state(cands[i], self.p.sigma_m, self.p.sigma_v0)
            m = self.model
            self.kf = KalmanFilter(m.D, m.M, m.Sd, m.Sm, x0, P0)
            self.state = TrackState.TENTATIVE
            self.hits, self.dark, self.still, self.anchor = 1, 0, 0, None
            return TrackStep(self.frame, self.state, len(cands), cands[i], i, None, None, None,
                             self.kf.x.copy(), self.kf.P.copy(), None, 0, new_track=True)

        # ---- com trilha: predição + gating + correção
        kf = self.kf
        if self.dark > 0:
            # no escuro, a velocidade "esquece" aos poucos: uma velocidade errada (de uma
            # associação ruim) não leva a predição para fora do campo
            kf.x[2:] *= self.p.dark_damping
        x_pred, _ = kf.predict()
        x_pred = x_pred.copy()
        z_pred, S = kf.predicted_measurement(Sm)
        S_gate = self._gate_S(S)
        i, d2 = self._associate(cands, sc, z_pred, S_gate)

        lost, reason = False, ""
        if i is not None:
            kf.correct(cands[i], Sm)
            speed = float(np.hypot(kf.x[2], kf.x[3]))
            if speed > self.p.max_speed:   # limite físico: a bola não passa disso
                kf.x[2:] *= self.p.max_speed / speed
            self.hits += 1
            self.dark = 0
            self.state = TrackState.TRACKING if self.hits >= self.p.confirm_hits else TrackState.TENTATIVE
            z = cands[i]
            # parada = a posição não sai de um raio pequeno (a velocidade estimada é ruidosa)
            if self.anchor is not None and np.linalg.norm(kf.x[:2] - self.anchor) < self.p.static_radius:
                self.still += 1
            else:
                self.anchor, self.still = kf.x[:2].copy(), 0
            if self.still > self.p.static_max_s * self.p.fps and len(cands) > 1:
                lost, reason = True, "parada"
                self.clutter.mark(z)
        else:
            z = None
            self.dark += 1
            if self.state == TrackState.TENTATIVE:
                if self.dark > self.p.tentative_max_dark:
                    lost, reason = True, "tentativa"
            else:
                self.state = TrackState.DARK
                if self.dark > self.p.max_dark:
                    lost, reason = True, "no_escuro"

        self.clutter.update(cands if i is None else np.delete(cands, i, axis=0))
        step = TrackStep(self.frame, self.state, len(cands), z, i, x_pred, z_pred, S_gate,
                         kf.x.copy(), kf.P.copy(), d2, self.dark, lost=lost, lost_reason=reason)
        if lost:
            self._drop()
            step.state = TrackState.SEARCHING
        return step
