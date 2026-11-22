"""Geometria do campo: calibração por marcos, homografia do plano da bola e jacobiano."""
import cv2
import numpy as np

from src.field import (BALL_RADIUS, Calibration, FieldModel, Landmark, LandmarkSet, _look_at,
                       calibrate, iphone15_video_K)
from src.minimap import Minimap


def synthetic_setup(C=(12.5, 4.0, 1.2), target=(5.0, 2.0, 0.5), f=1700.0, h=2.0):
    K = iphone15_video_K()
    K[0, 0] = K[1, 1] = f
    rvec, tvec = _look_at(np.array(C, float), np.array(target, float))

    def proj(P):
        uv, _ = cv2.projectPoints(np.array(P, float).reshape(-1, 3), rvec, tvec, K, None)
        return uv.reshape(-1, 2)

    pts = [Landmark("base8", tuple(proj([8, 0, 0])[0]), [8.0, 0.0, 0.0]),
           Landmark("base5", tuple(proj([5, 0, 0])[0]), [5.0, 0.0, 0.0]),
           Landmark("topo5", tuple(proj([5, 0, h])[0]), [5.0, 0.0, "h"]),
           Landmark("eixo8", tuple(proj([8, 0, 1.3])[0]), [8.0, 0.0, None])]
    for y in (3.0, 6.0, 9.0, 12.0):
        pts.append(Landmark(f"lat{y}", tuple(proj([0, y, 0])[0]), [0.0, None, 0.0]))
    return LandmarkSet("sint", 0, (1920, 1080), pts), K, proj


def test_calibrate_recovers_pose_height_and_focal():
    lm, K, _ = synthetic_setup()
    K0 = iphone15_video_K()                    # f errado de propósito (1476 contra 1700)
    c = calibrate(lm, K0, fit_f=True)
    assert c.rms_px < 0.5
    assert abs(c.K[0, 0] - 1700) < 20
    assert abs(c.shared["h"] - 2.0) < 0.05
    assert np.linalg.norm(c.camera_center - np.array([12.5, 4.0, 1.2])) < 0.15


def test_ball_plane_homography_and_jacobian():
    lm, K, proj = synthetic_setup()
    c = calibrate(lm, K)
    fm = FieldModel(c, 640)
    P = np.array([6.0, 5.0, BALL_RADIUS])                 # centro de uma bola no chão
    uv = proj(P)[0] / 3.0                                  # px de processamento (640)
    XY = fm.to_world(uv)[0]
    assert np.allclose(XY, P[:2], atol=0.02)
    # jacobiano analítico x diferenças finitas
    J = fm.jacobian(uv)
    e = 1e-3
    num = np.column_stack([(fm.to_world(uv + [e, 0])[0] - fm.to_world(uv - [e, 0])[0]) / (2 * e),
                           (fm.to_world(uv + [0, e])[0] - fm.to_world(uv - [0, e])[0]) / (2 * e)])
    assert np.allclose(J, num, rtol=1e-4, atol=1e-7)
    # um px vale mais metros longe da câmera
    far = fm.to_image(np.array([[1.0, 12.0]]))[0]
    near = fm.to_image(np.array([[9.0, 3.0]]))[0]
    assert np.abs(np.linalg.det(fm.jacobian(far))) > np.abs(np.linalg.det(fm.jacobian(near)))
    # tamanho esperado da bola ~ f·0,22/distância
    d = fm.expected_diameter_px(uv)
    dist = np.linalg.norm(P - c.camera_center)
    assert abs(d - K[0, 0] / 3 * 0.22 / dist) < 0.2


def test_calibration_roundtrip_and_minimap(tmp_path):
    lm, K, _ = synthetic_setup()
    c = calibrate(lm, K)
    p = tmp_path / "f.json"
    c.save(p)
    c2 = Calibration.load(p)
    assert np.allclose(c.H_image_to_world, c2.H_image_to_world)
    mm = Minimap(FieldModel(c2, 640), 360, (640, 360))
    img = mm.draw(np.array([6.5, 3.0]), np.array([6.4, 3.1]), 12.0)
    assert img.shape == (360, mm.w, 3)
