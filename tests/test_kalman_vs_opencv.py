"""Validação: o nosso filtro e o cv2.KalmanFilter, com as mesmas matrizes e a mesma
sequência de medições (incluindo quadros sem medição), devem dar o mesmo resultado."""
import cv2
import numpy as np
import pytest

from src.kalman import KalmanFilter, constant_velocity, init_state
from src.synthetic import SyntheticConfig, generate


def make_opencv(model, x0, P0):
    kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_64F)
    kf.transitionMatrix = model.D.copy()
    kf.measurementMatrix = model.M.copy()
    kf.processNoiseCov = model.Sd.copy()
    kf.measurementNoiseCov = model.Sm.copy()
    kf.statePost = x0.reshape(4, 1).copy()
    kf.errorCovPost = P0.copy()
    return kf


@pytest.mark.parametrize("sigma_a", [5.0, 200.0, 3000.0])
def test_matches_opencv(sigma_a):
    seq = generate(SyntheticConfig(n_frames=250, occlusions=[(60, 85), (150, 160)],
                                   miss_prob=0.1, seed=3))
    model = constant_velocity(seq.dt, sigma_a=sigma_a, sigma_m=1.5)
    first = int(np.flatnonzero(~np.isnan(seq.measured[:, 0]))[0])
    x0, P0 = init_state(seq.measured[first], 1.5, 300.0)

    ours = KalmanFilter(model.D, model.M, model.Sd, model.Sm, x0, P0)
    cv = make_opencv(model, x0, P0)

    for t in range(first + 1, seq.config.n_frames):
        xp, Pp = ours.predict()
        cv_xp = cv.predict()
        np.testing.assert_allclose(xp, cv_xp.ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(Pp, cv.errorCovPre, rtol=1e-9, atol=1e-9)
        z = seq.measured[t]
        if not np.isnan(z[0]):
            xc, Pc = ours.correct(z)
            cv_xc = cv.correct(z.reshape(2, 1).astype(np.float64))
            np.testing.assert_allclose(xc, cv_xc.ravel(), rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(Pc, cv.errorCovPost, rtol=1e-9, atol=1e-9)
        else:
            # sem medição: no OpenCV a predição também vira o estado "post"
            np.testing.assert_allclose(ours.x, cv.statePost.ravel(), rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(ours.P, cv.errorCovPost, rtol=1e-9, atol=1e-9)


def test_filter_reduces_noise():
    """Com movimento suave, a estimativa filtrada fica mais perto da verdade que a medição."""
    seq = generate(SyntheticConfig(n_frames=400, kick_every=(120, 200), sigma_m=2.0, seed=5,
                                   area=(4000.0, 4000.0), start=(2000.0, 2000.0),
                                   speed_range=(40.0, 120.0)))
    model = constant_velocity(seq.dt, sigma_a=100.0, sigma_m=2.0)
    x0, P0 = init_state(seq.measured[0], 2.0, 300.0)
    kf = KalmanFilter(model.D, model.M, model.Sd, model.Sm, x0, P0)
    est = [x0[:2]]
    for t in range(1, seq.config.n_frames):
        kf.predict()
        kf.correct(seq.measured[t])
        est.append(kf.x[:2].copy())
    est = np.array(est)[30:]
    truth = seq.truth[30:, :2]
    raw = seq.measured[30:]
    err_raw = np.mean(np.linalg.norm(raw - truth, axis=1))
    err_kf = np.mean(np.linalg.norm(est - truth, axis=1))
    assert err_kf < 0.8 * err_raw, (err_kf, err_raw)


def test_constant_velocity_matrices():
    m = constant_velocity(0.1, sigma_a=2.0, sigma_m=3.0)
    assert m.D[0, 2] == pytest.approx(0.1) and m.D[1, 3] == pytest.approx(0.1)
    assert np.allclose(m.Sd, m.Sd.T)
    assert np.all(np.linalg.eigvalsh(m.Sd) >= -1e-12)
    assert m.Sd[2, 2] == pytest.approx(4.0 * 0.01)
    assert m.Sd[0, 0] == pytest.approx(4.0 * 0.1 ** 4 / 4)
    assert np.allclose(m.Sm, 9.0 * np.eye(2))
