"""A estabilização deve recuperar um movimento de câmera conhecido (sequência sintética)."""
import cv2
import numpy as np

from src.stabilize import Stabilizer


def textured_scene(w=900, h=560, seed=1):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(h // 8, w // 8, 3), dtype=np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    for _ in range(80):  # alguns cantos fortes
        x, y = rng.integers(0, w - 40), rng.integers(0, h - 40)
        cv2.rectangle(img, (int(x), int(y)), (int(x + 25), int(y + 15)),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def camera_pose(t):
    """Homografia cena -> quadro: pequena rotação + translação, como uma câmera na mão."""
    ang = np.deg2rad(1.5 * np.sin(t / 15))
    tx, ty = -130 + 25 * np.sin(t / 10), -100 + 15 * np.cos(t / 13)
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.0]])


def test_recovers_known_camera_motion():
    scene = textured_scene()
    w, h = 640, 360
    stab = Stabilizer()
    pts = np.float64([[50, 50], [600, 60], [320, 180], [80, 320], [590, 330]])
    errors = []
    G0 = camera_pose(0)
    for t in range(90):
        G = camera_pose(t)
        frame = cv2.warpPerspective(scene, G, (w, h))
        st = stab.update(frame)
        # verdade: quadro t -> quadro 0 = G0 · G_t^-1
        H_true = G0 @ np.linalg.inv(G)
        est = Stabilizer.apply(st.H, pts)
        ref = Stabilizer.apply(H_true, pts)
        errors.append(np.linalg.norm(est - ref, axis=1).max())
    assert max(errors) < 2.0, f"erro máximo {max(errors):.2f} px"
