"""Semantic checks for the public scene observation adapter; no simulator/model."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'source'))
from s_entities import layout_of, state_entities
import u_wm


def test_scene_object_identity_and_joint_changes():
    L = layout_of('scene-play-v0', 40)
    L.update(w_obs=0.4, loc_uv=[[30., 36.], [34., 36.]],
             handle_affine=[[[0., 25.], [4., 15.]], [[0., 28.], [5., 20.]]])
    ob = np.zeros((2, 40), np.float32)
    ob[:, 19:22] = [.5, -.2, .2]
    ob[:, 28:30] = [1., 0.]
    ob[:, 32:34] = [0., 1.]
    ob[:, 36] = [-1., -2.]
    ob[:, 38] = [0., 1.]
    S, _ = state_entities(ob, L)
    assert S.shape == (2, 5, 6)
    np.testing.assert_allclose(S[:, 0, :3], [[36., 30.4, .05]] * 2, atol=1e-6)
    np.testing.assert_allclose(S[:, 1, 2:5], 0.)
    np.testing.assert_allclose(S[:, 2, 2:5], 1.)
    np.testing.assert_allclose(S[:, 3, :2], [[25., 11.], [25., 7.]])
    np.testing.assert_allclose(S[:, 4, :2], [[28., 20.], [28., 25.]])
    # Velocities/proprioception must not change persistent object state.
    ob2 = ob.copy()
    ob2[:, :19] = 1.
    ob2[:, [31, 35, 37, 39]] = 123.
    np.testing.assert_array_equal(state_entities(ob2, L)[0], S)


def test_finite_support_preserves_continuous_positions():
    assert hasattr(u_wm, 'finite_rest_support'), 'train-derived finite support is not yet integrated'
    S = np.zeros((100, 2, 6), np.float32)
    S[:, 0, 0] = np.linspace(20., 40., 100)
    S[:, 1, 2:5] = (np.arange(100) % 2)[:, None]
    support = u_wm.finite_rest_support(S)
    assert support[0][0] is None
    assert support[1][2] == [0., 1.]


if __name__ == '__main__':
    test_scene_object_identity_and_joint_changes()
    test_finite_support_preserves_continuous_positions()
    print('SCENE_PUBLIC_STATE_SEMANTICS_OK')
