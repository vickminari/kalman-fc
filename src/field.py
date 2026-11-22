"""Geometria do campo: intrínsecos da câmera, pose a partir de marcos e homografia do chão.

Sistema do MUNDO (metros), definido pelo gol filmado:
    X  ao longo da linha do gol, a partir de um escanteio (X = 0) até o outro (X = 13);
       as traves ficam em X = 5 e X = 8 (medidas do campo: 5 m + 3 m + 5 m)
    Y  perpendicular à linha do gol, entrando no campo (Y = 0 na linha do gol)
    Z  para cima (Z = 0 no gramado)

Projeção (modelo pinhole):   s·[u, v, 1]ᵀ = K · (R · P + t)

Os vídeos de teste não têm marcos medidos fora da linha do gol, e só com pontos alinhados
não se obtém a homografia do chão. Com a matriz K conhecida (calibração do Gabriel), usamos
também estruturas cuja GEOMETRIA é conhecida, mesmo sem medida:
- os eixos das traves são verticais (X e Y fixos, Z livre);
- o topo das traves fica na altura h do travessão (h desconhecido, compartilhado);
- a base da parede lateral é perpendicular à linha do gol (X fixo, Y livre).

A pose (R, t), as incógnitas compartilhadas (h, ...) e as coordenadas livres de cada ponto
são estimadas juntas por mínimos quadrados no erro de reprojeção. Com a pose, a homografia
do chão (Z = 0) sai direto:

    H_mundo→imagem = K · [r1  r2  t]        H_imagem→mundo = H⁻¹

O jacobiano J de H_imagem→mundo num ponto leva a covariância da medição de pixels para
metros:  Σm_campo = J Σm_px Jᵀ  (o "filtro em metros", F2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

# ------------------------------------------------------------------ campo (medidas)
GOAL_LINE_LENGTH = 13.0          # escanteio a escanteio (m)
POSTS_X = (5.0, 8.0)             # traves (m)
BALL_DIAMETER = 0.22             # bola de futebol (m), para o tamanho esperado
BALL_RADIUS = BALL_DIAMETER / 2

# ------------------------------------------------------------------ câmera
# Calibração de Zhang do iPhone 15 (câmera principal) feita por Gabriel Lopes Bastos
# (Atividade Prática 01): fotos HEIC 4284 x 5712 (retrato), RMSE final 1,60 px.
IPHONE15_PHOTO = dict(width=4284, height=5712, fx=4389.467, fy=4393.468,
                      cx=2144.224, cy=2846.219, skew=7.969, k1=0.154811, k2=-0.540099)
# ARKit (sistema), quadros 1920 x 1440: fx = fy ≈ 1478 px — referência para o vídeo.


def iphone15_video_K(width: int = 1920, height: int = 1080) -> np.ndarray:
    """K aproximada para o vídeo 16:9 a partir da calibração das fotos.

    Hipóteses (verificáveis estimando f na própria cena, ver `calibrate(fit_f=True)`):
    1. o vídeo usa a largura inteira do sensor (foto deitada: 5712 px) e corta 16:9 no centro;
    2. a foto em retrato vira paisagem por rotação de 90°: fx_paisagem = fy_retrato e vice-versa;
    3. o ponto principal fica no centro (a diferença medida pelo Gabriel é < 1%).
    Resultado em 1920 x 1080: f ≈ 1476 px (o ARKit dá ≈ 1478 px para quadros de 1920 de largura).
    """
    p = IPHONE15_PHOTO
    s = width / p["height"]                     # 1920 / 5712
    fx, fy = p["fy"] * s, p["fx"] * s
    return np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1.0]])


def scale_K(K: np.ndarray, s: float) -> np.ndarray:
    """K para a imagem redimensionada por s (ex.: 1920 -> 640: s = 1/3)."""
    S = np.diag([s, s, 1.0])
    return S @ K


# ------------------------------------------------------------------ marcos
@dataclass
class Landmark:
    name: str
    xy: tuple[float, float]                 # pixels da imagem de calibração
    world: list                              # [X, Y, Z]: número, None (livre) ou "nome" (compartilhado)


@dataclass
class LandmarkSet:
    video: str
    frame: int
    image_size: tuple[int, int]
    points: list[Landmark]
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "LandmarkSet":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        pts = [Landmark(p["name"], tuple(p["xy"]), list(p["world"])) for p in d["points"]]
        return cls(d["video"], d.get("frame", 0), tuple(d["image_size"]), pts, d.get("notes", ""))

    def save(self, path: str | Path) -> None:
        d = dict(video=self.video, frame=self.frame, image_size=list(self.image_size), notes=self.notes,
                 points=[dict(name=p.name, xy=[round(float(v), 1) for v in p.xy], world=p.world)
                         for p in self.points])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


@dataclass
class Calibration:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    shared: dict[str, float]
    residuals_px: dict[str, float]
    rms_px: float
    image_size: tuple[int, int]
    camera_center: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def H_world_to_image(self) -> np.ndarray:
        return self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])

    @property
    def H_image_to_world(self) -> np.ndarray:
        H = np.linalg.inv(self.H_world_to_image)
        return H / H[2, 2]

    def project(self, P: np.ndarray) -> np.ndarray:
        """Projeta pontos 3D (N, 3) na imagem de calibração."""
        P = np.atleast_2d(P)
        cam = (self.R @ P.T).T + self.t
        uvw = (self.K @ cam.T).T
        return uvw[:, :2] / uvw[:, 2:3]

    def to_dict(self) -> dict:
        return dict(K=self.K.tolist(), R=self.R.tolist(), t=self.t.tolist(), shared=self.shared,
                    residuals_px=self.residuals_px, rms_px=self.rms_px,
                    image_size=list(self.image_size), camera_center=self.camera_center.tolist(),
                    H_image_to_world=self.H_image_to_world.tolist())

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(np.array(d["K"]), np.array(d["R"]), np.array(d["t"]), d["shared"],
                   d["residuals_px"], d["rms_px"], tuple(d["image_size"]), np.array(d["camera_center"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _look_at(C: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rvec, tvec) de uma câmera em C olhando para `target`, com Z do mundo para cima."""
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel(), -R @ C


def calibrate(lm: LandmarkSet, K: np.ndarray, fit_f: bool = False,
              inits: list[tuple[np.ndarray, np.ndarray]] | None = None,
              priors: dict[str, tuple[float, float]] | None = None,
              prior_px: float = 2.0) -> Calibration:
    """Pose da câmera (e incógnitas dos marcos) por mínimos quadrados no erro de reprojeção.

    `priors`: conhecimento aproximado, como resíduo extra (valor − média)/desvio · `prior_px`.
    Chaves: "camera_height" (altura da câmera, m) ou o nome de uma incógnita compartilhada
    (ex.: "d"). Serve quando os marcos sozinhos não determinam bem a pose (vídeo do centro).
    """
    priors = priors or {}
    shared_names = sorted({w for p in lm.points for w in p.world if isinstance(w, str)})
    free_slots = [(i, k) for i, p in enumerate(lm.points) for k, w in enumerate(p.world) if w is None]
    uv = np.array([p.xy for p in lm.points], float)

    def unpack(v):
        rvec, tvec = v[:3], v[3:6]
        i = 6
        f = None
        if fit_f:
            f, i = v[6], 7
        sh = dict(zip(shared_names, v[i:i + len(shared_names)]))
        i += len(shared_names)
        free = v[i:]
        P = np.zeros((len(lm.points), 3))
        for j, p in enumerate(lm.points):
            for k, w in enumerate(p.world):
                if isinstance(w, (int, float)):
                    P[j, k] = w
                elif isinstance(w, str):
                    P[j, k] = sh[w]
        for (j, k), val in zip(free_slots, free):
            P[j, k] = val
        KK = K.copy()
        if f is not None:
            KK[0, 0] = KK[1, 1] = f
        return rvec, tvec, KK, sh, P

    def resid(v):
        rvec, tvec, KK, sh, P = unpack(v)
        proj, _ = cv2.projectPoints(P, rvec, tvec, KK, None)
        r = list((proj.reshape(-1, 2) - uv).ravel())
        for key, (mu, sd) in priors.items():
            if key == "camera_height":
                R, _ = cv2.Rodrigues(rvec)
                val = float((-R.T @ tvec)[2])
            else:
                val = float(sh[key])
            r.append((val - mu) / sd * prior_px)
        return np.array(r)

    if inits is None:
        inits = []
        # 1) PnP com os marcos sem coordenada livre (incógnitas compartilhadas valendo 2,0)
        known = [j for j, p in enumerate(lm.points) if all(w is not None for w in p.world)]
        if len(known) >= 4:
            P0 = np.array([[2.0 if isinstance(w, str) else w for w in lm.points[j].world] for j in known])
            ok, rv, tv = cv2.solvePnP(P0, uv[known], K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                inits.append((rv.ravel(), tv.ravel()))
        # 2) câmeras plausíveis em volta do campo, a ~1,5 m de altura, olhando para o gol
        for C in ([16, 6, 1.5], [-3, 6, 1.5], [6.5, 12, 1.5], [6.5, 8, 1.6], [14, 10, 1.5], [-1, 10, 1.5]):
            inits.append(_look_at(np.array(C, float), np.array([6.5, 1.0, 0.5])))
    def initial_unknowns(rvec0, tvec0):
        """Chute das coordenadas livres e compartilhadas: retroprojeta cada ponto da imagem
        no plano do chão (ou na reta vertical do marco) usando a pose inicial."""
        R0, _ = cv2.Rodrigues(np.asarray(rvec0, float))
        C0 = -R0.T @ np.asarray(tvec0, float)
        Kinv = np.linalg.inv(K)
        guess: dict[tuple[int, int], float] = {}
        for j, p in enumerate(lm.points):
            ray = R0.T @ (Kinv @ np.array([*p.xy, 1.0]))
            fixed = {k: w for k, w in enumerate(p.world) if isinstance(w, (int, float))}
            if fixed.get(2) == 0.0 and abs(ray[2]) > 1e-9:        # ponto no chão
                s = -C0[2] / ray[2]
                Q = C0 + s * ray
            elif 0 in fixed and 1 in fixed:                       # ponto numa reta vertical
                # ponto da reta (X, Y, z) mais próximo do raio
                A = np.array([fixed[0], fixed[1], 0.0])
                u = np.array([0, 0, 1.0])
                w0 = C0 - A
                a, b, c = ray @ ray, ray @ u, u @ u
                d_, e = ray @ w0, u @ w0
                den = a * c - b * b
                tz = (a * e - b * d_) / den if abs(den) > 1e-12 else 1.0
                Q = A + tz * u
            else:
                Q = np.array([6.5, 4.0, 1.0])
            for k in range(3):
                guess[(j, k)] = float(Q[k])
        shared0 = []
        for nm in shared_names:
            vals = [guess[(j, k)] for j, p in enumerate(lm.points) for k, w in enumerate(p.world) if w == nm]
            shared0.append(float(np.median(vals)) if vals else 2.0)
        free0 = [guess[(j, k)] for j, k in free_slots]
        return shared0, free0

    best = None
    for rvec0, tvec0 in inits:
        shared0, free0 = initial_unknowns(rvec0, tvec0)
        v0 = list(rvec0) + list(tvec0) + ([K[0, 0]] if fit_f else []) + shared0 + free0
        try:
            r = least_squares(resid, np.array(v0, float), method="lm", max_nfev=20000)
        except Exception:
            continue
        rvec, tvec, KK, sh, P = unpack(r.x)
        R, _ = cv2.Rodrigues(rvec)
        depth_ok = np.all((R @ P.T).T[:, 2] + tvec[2] > 0)   # tudo na frente da câmera
        C = -R.T @ tvec
        sane = (np.linalg.norm(C) < 100 and 0 < C[2] < 20 and np.all(np.abs(P) < 200)
                and all(abs(v) < 50 for v in sh.values()))
        if depth_ok and sane and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        raise RuntimeError("calibração não convergiu")
    rvec, tvec, KK, sh, P = unpack(best.x)
    R, _ = cv2.Rodrigues(rvec)
    res = best.fun[:2 * len(lm.points)].reshape(-1, 2)
    per = {p.name: round(float(np.hypot(*res[i])), 2) for i, p in enumerate(lm.points)}
    return Calibration(KK, R, tvec, {k: float(v) for k, v in sh.items()}, per,
                       float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))), lm.image_size,
                       camera_center=-R.T @ tvec)


# ------------------------------------------------------------------ uso no pipeline
class FieldModel:
    """Homografia imagem (px de processamento do quadro de referência) <-> campo (m).

    A homografia é a do plano horizontal Z = `plane_z`. Para a BOLA usamos Z = raio
    (0,11 m): o centro detectado da bola está a essa altura, e projetá-lo no gramado
    (Z = 0) o jogaria para longe da câmera, sobretudo no fundo do campo.

        H_mundo→imagem(z) = K · [r1  r2  (z·r3 + t)]
    """

    def __init__(self, calib: Calibration, proc_width: int, plane_z: float = BALL_RADIUS):
        self.calib = calib
        self.plane_z = plane_z
        s = proc_width / calib.image_size[0]
        S = np.diag([s, s, 1.0])
        R, t = calib.R, calib.t
        H = calib.K @ np.column_stack([R[:, 0], R[:, 1], plane_z * R[:, 2] + t])
        self.H_w2i = S @ H                                 # mundo (plano z) -> px de processamento
        self.H_i2w = np.linalg.inv(self.H_w2i)
        self.scale = s

    def in_front(self, XY: np.ndarray) -> np.ndarray:
        """True para pontos do plano que estão na frente da câmera."""
        XY = np.atleast_2d(XY)
        P = np.column_stack([XY, np.full(len(XY), self.plane_z)])
        return (self.calib.R @ P.T).T[:, 2] + self.calib.t[2] > 0

    def visible_region(self, width: int, height: int, max_dist: float = 40.0) -> np.ndarray:
        """Polígono (no campo, m) do chão visível no quadro de referência."""
        border = np.concatenate([
            np.stack([np.linspace(0, width, 40), np.full(40, height)], 1),
            np.stack([np.full(40, width), np.linspace(height, 0, 40)], 1),
            np.stack([np.linspace(width, 0, 40), np.zeros(40)], 1),
            np.stack([np.zeros(40), np.linspace(0, height, 40)], 1)])
        a = np.column_stack([border, np.ones(len(border))]) @ self.H_i2w.T
        ok = a[:, 2] * np.sign(np.linalg.det(self.H_i2w)) != 0
        W = a[:, :2] / a[:, 2:3]
        C = self.calib.camera_center[:2]
        ok &= self.in_front(W) & (np.linalg.norm(W - C, axis=1) < max_dist)
        return W[ok]

    def to_world(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_i2w).reshape(-1, 2)

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_w2i).reshape(-1, 2)

    def jacobian(self, uv: np.ndarray) -> np.ndarray:
        """∂(X, Y)/∂(u, v) da homografia imagem -> mundo no ponto uv (2x2, m/px)."""
        H = self.H_i2w
        u, v = float(uv[0]), float(uv[1])
        a = H @ np.array([u, v, 1.0])
        w = a[2]
        J = np.empty((2, 2))
        for i in range(2):
            J[i, 0] = (H[i, 0] * w - a[i] * H[2, 0]) / w ** 2
            J[i, 1] = (H[i, 1] * w - a[i] * H[2, 1]) / w ** 2
        return J

    def expected_diameter_px(self, uv: np.ndarray) -> float:
        """Diâmetro esperado (px) de uma bola de 0,22 m apoiada no chão no ponto uv."""
        Xw = self.to_world(uv)[0]
        C = self.calib.camera_center
        P = np.array([Xw[0], Xw[1], BALL_DIAMETER / 2])
        dist = np.linalg.norm(P - C)
        f = self.calib.K[0, 0] * self.scale
        return float(f * BALL_DIAMETER / dist)
