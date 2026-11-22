"""Compensação do movimento da câmera (câmera na mão).

Ideia: escolher um quadro de referência e, para cada quadro t, estimar a homografia
H_t que leva pixels do quadro t para pixels da referência. Com isso:

- a bola medida no quadro t vira uma medição em "coordenadas da referência", onde o
  fundo é fixo, e o filtro de Kalman roda lá (o movimento da câmera não vira
  "velocidade da bola");
- a homografia do campo (referência -> metros) continua valendo para todos os quadros:
  H_campo(t) = H_ref->campo · H_t.

Uma homografia é o modelo certo aqui. A câmera na mão quase só gira, e para uma
rotação pura a relação entre dois quadros é exatamente uma homografia, para qualquer
ponto da cena. Para pontos do chão ela vale até com translação.

Estimativa:
1. quadro a quadro: cantos (Shi-Tomasi) rastreados por fluxo óptico de Lucas-Kanade
   com checagem ida e volta, e homografia por RANSAC. É rápido, mas o erro acumula (deriva);
2. correção da deriva: a cada `correct_every` quadros, o quadro atual é casado
   diretamente (ORB + RANSAC) com o quadro-chave de maior sobreposição, cuja
   homografia para a referência já é conhecida;
3. quando a câmera vira para uma região que nenhum quadro-chave cobre bem, o quadro
   atual vira um novo quadro-chave.

Todas as homografias públicas estão em pixels do quadro de entrada (resolução de
processamento). Internamente o rastreamento roda em `work_width` px.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StabilizerParams:
    work_width: int = 640
    max_corners: int = 500
    min_corners: int = 150           # abaixo disso, detecta cantos novos
    lk_win: int = 21
    lk_levels: int = 3
    fb_max_err: float = 1.0          # erro máximo da checagem ida e volta (px)
    ransac_px: float = 2.0
    min_inliers: int = 40
    correct_every: int = 10          # quadros entre correções de deriva
    correct_below_inliers: int = 150 # corrige também quando o rastreamento enfraquece
    max_correction_px: float = 10.0  # correção maior que isso (com LK ok) é rejeitada
    min_inlier_spread: float = 0.2   # fecho convexo dos inliers ORB / área do quadro
    redetect_every: int = 10         # renova os cantos rastreados a cada N quadros
    agree_px: float = 3.0            # duas correções grandes seguidas que concordam são aceitas
    orb_features: int = 2000
    match_gate_px: float = 25.0      # casamentos ORB precisam concordar com a predição
    new_keyframe_overlap: float = 0.6
    max_keyframes: int = 60


@dataclass
class StabState:
    frame: int
    H: np.ndarray                    # quadro atual -> referência (px de entrada)
    ok: bool                         # estimativa quadro a quadro funcionou
    inliers: int
    corrected: bool = False          # houve correção por quadro-chave neste quadro
    correction_px: float = float("nan")  # quanto a correção moveu os cantos (px de entrada)
    keyframe: int = -1
    n_keyframes: int = 0


@dataclass
class _Keyframe:
    frame: int
    H: np.ndarray                    # quadro-chave -> referência (px de trabalho)
    kp: np.ndarray                   # pontos ORB (N x 2, px de trabalho)
    desc: np.ndarray


class Stabilizer:
    def __init__(self, params: StabilizerParams | None = None):
        self.p = params or StabilizerParams()
        self._orb_det = cv2.ORB_create(self.p.orb_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._prev: np.ndarray | None = None
        self._pts: np.ndarray | None = None
        self._H = np.eye(3)          # atual -> referência (px de trabalho)
        self._keyframes: list[_Keyframe] = []
        self._S = np.eye(3)          # entrada -> trabalho
        self._size = (0, 0)          # (w, h) de trabalho
        self._idx = -1
        self._rejected_delta: np.ndarray | None = None
        self.history: list[StabState] = []

    # ------------------------------------------------------------------ util
    def _to_work(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        s = self.p.work_width / w if w > self.p.work_width else 1.0
        self._S = np.diag([s, s, 1.0])
        size = (int(round(w * s)), int(round(h * s)))
        self._size = size
        small = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA) if s != 1.0 else frame_bgr
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _public(self, H_work: np.ndarray) -> np.ndarray:
        S, Si = self._S, np.linalg.inv(self._S)
        return Si @ H_work @ S

    def _corners(self) -> np.ndarray:
        w, h = self._size
        return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    def _overlap(self, H_a: np.ndarray, H_b: np.ndarray) -> float:
        """Fração da área do quadro A (projetado na referência) coberta pelo quadro B."""
        a = cv2.perspectiveTransform(self._corners(), H_a).reshape(-1, 2)
        b = cv2.perspectiveTransform(self._corners(), H_b).reshape(-1, 2)
        area_a = cv2.contourArea(a)
        if area_a <= 0:
            return 0.0
        inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
        return float(inter / area_a)

    def _orb_features(self, gray: np.ndarray):
        kp, desc = self._orb_det.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
        return pts, desc

    def _add_keyframe(self, gray: np.ndarray, H: np.ndarray) -> None:
        pts, desc = self._orb_features(gray)
        if desc is None or len(pts) < self.p.min_inliers:
            return
        self._keyframes.append(_Keyframe(self._idx, H.copy(), pts, desc))
        if len(self._keyframes) > self.p.max_keyframes:
            self._keyframes.pop(1)   # nunca descarta a referência

    def _detect_corners(self, gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(gray, self.p.max_corners, 0.01, 8, blockSize=7)

    # ------------------------------------------------------------------ passos
    def _frame_to_frame(self, gray: np.ndarray) -> tuple[np.ndarray | None, int]:
        """Homografia anterior -> atual por LK com checagem ida e volta."""
        p = self.p
        # renova os pontos periodicamente: sem isso o conjunto rastreado pode "colapsar"
        # sobre um jogador grande que ficou parado por alguns quadros
        if (self._pts is None or len(self._pts) < p.min_corners
                or self._idx % p.redetect_every == 0):
            self._pts = self._detect_corners(self._prev)
        if self._pts is None or len(self._pts) < 8:
            return None, 0
        lk = dict(winSize=(p.lk_win, p.lk_win), maxLevel=p.lk_levels,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._prev, gray, self._pts, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev, nxt, None, **lk)
        fb = np.linalg.norm((self._pts - back).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < p.fb_max_err)
        a, b = self._pts[good], nxt[good]
        if len(a) < 8:
            self._pts = None
            return None, 0
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, p.ransac_px)
        if H is None:
            self._pts = None
            return None, 0
        inl = inl.ravel().astype(bool)
        self._pts = b[inl].reshape(-1, 1, 2)   # continua rastreando só os inliers (fundo)
        return H, int(inl.sum())

    def _correct(self, gray: np.ndarray) -> tuple[np.ndarray | None, int, float]:
        """Casa o quadro atual com o melhor quadro-chave. Retorna (H atual->ref, kf, deslocamento)."""
        p = self.p
        if not self._keyframes:
            return None, -1, float("nan")
        best = max(self._keyframes, key=lambda k: self._overlap(self._H, k.H))
        pts, desc = self._orb_features(gray)
        if desc is None or len(pts) < p.min_inliers:
            return None, -1, float("nan")
        matches = self._bf.knnMatch(desc, best.desc, k=2)
        src, dst = [], []
        H_pred = np.linalg.inv(best.H) @ self._H   # atual -> quadro-chave (predição)
        pred = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_pred).reshape(-1, 2)
        for m in matches:
            if len(m) < 2 or m[0].distance > 0.8 * m[1].distance:
                continue
            q, t = m[0].queryIdx, m[0].trainIdx
            if np.linalg.norm(pred[q] - best.kp[t]) > p.match_gate_px:
                continue
            src.append(pts[q])
            dst.append(best.kp[t])
        if len(src) < p.min_inliers:
            return None, best.frame, float("nan")
        src, dst = np.float32(src), np.float32(dst)
        H_ck, inl = cv2.findHomography(src, dst, cv2.RANSAC, p.ransac_px)
        if H_ck is None or inl.sum() < p.min_inliers:
            return None, best.frame, float("nan")
        # os inliers precisam cobrir boa parte do quadro; casamentos concentrados (num
        # jogador ou num pedaço de grama) geram homografias distorcidas
        hull = cv2.convexHull(src[inl.ravel().astype(bool)])
        w, h = self._size
        if cv2.contourArea(hull) < p.min_inlier_spread * w * h:
            return None, best.frame, float("nan")
        H_new = best.H @ H_ck
        c = self._corners()
        d = np.linalg.norm(cv2.perspectiveTransform(c, H_new) - cv2.perspectiveTransform(c, self._H),
                           axis=2).mean() / self._S[0, 0]
        return H_new, best.frame, float(d)

    # ------------------------------------------------------------------ API
    def update(self, frame_bgr: np.ndarray) -> StabState:
        """Processa o próximo quadro e devolve o estado (H atual -> referência)."""
        self._idx += 1
        gray = self._to_work(frame_bgr)
        if self._prev is None:
            self._prev = gray
            self._H = np.eye(3)
            self._add_keyframe(gray, self._H)
            st = StabState(self._idx, self._public(self._H), True, 0, n_keyframes=len(self._keyframes))
            self.history.append(st)
            return st

        H_pc, inliers = self._frame_to_frame(gray)
        ok = H_pc is not None and inliers >= self.p.min_inliers
        if ok:
            self._H = self._H @ np.linalg.inv(H_pc)

        st = StabState(self._idx, self._public(self._H), ok, inliers)
        if (not ok or self._idx % self.p.correct_every == 0
                or inliers < self.p.correct_below_inliers):
            H_new, kf, d = self._correct(gray)
            st.keyframe = kf
            # com o rastreamento saudável, a deriva em poucos quadros é pequena: uma correção
            # grande indica casamento errado, não deriva
            # ...a menos que duas propostas seguidas concordem entre si (deriva real acumulada)
            if H_new is not None and ok and d > self.p.max_correction_px:
                delta = H_new @ np.linalg.inv(self._H)   # correção no plano da referência
                prev = self._rejected_delta
                self._rejected_delta = delta
                c = self._corners()
                agree = prev is not None and np.linalg.norm(
                    cv2.perspectiveTransform(c, delta) - cv2.perspectiveTransform(c, prev),
                    axis=2).mean() / self._S[0, 0] < self.p.agree_px
                if not agree:
                    H_new = None
                    st.correction_px = d   # registrada, mas rejeitada
            else:
                self._rejected_delta = None
            if H_new is not None:
                self._rejected_delta = None
            if H_new is not None:
                self._H = H_new
                st.corrected, st.correction_px = True, d
                st.H = self._public(self._H)
                best_overlap = max(self._overlap(self._H, k.H) for k in self._keyframes)
                if best_overlap < self.p.new_keyframe_overlap:
                    self._add_keyframe(gray, self._H)
        st.n_keyframes = len(self._keyframes)
        self._prev = gray
        self.history.append(st)
        return st

    # pontos ------------------------------------------------------------
    @staticmethod
    def apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Aplica uma homografia a pontos (N x 2)."""
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
