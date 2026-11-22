"""Pistas novas dos candidatos: movimento compensado, LSS, blobs e score combinado."""
import cv2
import numpy as np

from src.blob_detector import LoGBlobDetector
from src.detector import Candidate
from src.lss import descriptor, to_lab
from src.motion_cue import MotionCue
from src.scoring import BallScorer
from tests.test_detector import make_scene


def test_motion_cue_separates_static_from_moving_with_camera_motion():
    base = make_scene(w=700, h=400)
    mc = MotionCue(gap=3, search=2)
    for t in range(4):
        scene = base.copy()
        cv2.circle(scene, (200, 300), 6, (235, 235, 235), -1)            # "folha" parada
        cv2.circle(scene, (300 + 8 * t, 250), 6, (235, 235, 235), -1)    # bola andando 8 px/quadro
        shift = 3 * t                                                    # câmera anda 3 px/quadro
        frame = scene[0:362, shift:shift + 640].copy()
        H = np.array([[1, 0, shift], [0, 1, 0], [0, 0, 1.0]])           # quadro -> referência
        mc.update(frame, H)
    xs = np.array([200 - 9, 300 + 24 - 9.0])                            # posições no último quadro
    ys = np.array([300.0, 250.0])
    m = mc.scores(xs, ys, np.array([12.0, 12.0]))
    assert m[0] < 10 and m[1] > 40, m


def test_lss_descriptor_is_scale_normalized():
    img = make_scene()
    cv2.circle(img, (200, 200), 5, (235, 235, 235), -1)
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    d1 = descriptor(to_lab(img), 200, 200, 10)
    d2 = descriptor(to_lab(big), 400, 400, 20)
    other = descriptor(to_lab(img), 400, 300, 10)                       # só grama
    assert d1 is not None and d2 is not None
    assert np.abs(d1 - d2).mean() < 0.1
    assert other is None or np.abs(d1 - other).mean() > np.abs(d1 - d2).mean()


def test_log_blob_finds_ball_and_its_scale():
    img = make_scene()
    cv2.circle(img, (320, 250), 6, (235, 235, 235), -1)
    det = LoGBlobDetector().detect(img)
    near = [c for c in det.candidates if np.hypot(c.x - 320, c.y - 250) < 3]
    assert near and 6 < near[0].diameter < 20


def test_scorer_roundtrip_and_filter(tmp_path):
    sc = BallScorer(features=["score", "size_fit"], mean=[0.5, -0.5], std=[0.2, 0.5],
                    weights=[0.0, 1.0, 2.0], fill=[0.5, -0.5], threshold=0.0,
                    size_prior=[np.log(10.0), 0.0], lss_template=[0.0] * 36)
    p = tmp_path / "m.json"
    sc.save(p)
    sc2 = BallScorer.load(p)
    good = Candidate(10, 10, np.pi * 25, 10, 10, 0.9, 0.9, 0.9)            # d = 10 = esperado
    bad = Candidate(10, 10, np.pi * 400, 40, 40, 0.3, 0.9, 0.9)            # d = 40, score baixo
    X = sc2.feature_matrix([good, bad], np.array([np.nan, np.nan]), None, 640)
    z = sc2.z(X)
    assert z[0] > sc2.threshold > z[1]
