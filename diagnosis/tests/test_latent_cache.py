import numpy as np
import torch

from data.latent_cache import LatentCache, REGIME_TO_ID


def test_roundtrip_6d_latent_plus_state(tmp_path):
    path = tmp_path / "mw__jepa_wm_metaworld.h5"
    T = 6
    z = torch.randn(T, 1, 4, 4, 8)          # (T, V, H, W, D)
    action = torch.randn(T - 1, 4)
    proprio = torch.randn(T, 4)
    state = torch.randn(T, 39)
    gripper = torch.rand(T)

    with LatentCache(path, mode="w") as c:
        c.write_trajectory("reach-v2/00000", z=z, action=action,
                           proprio=proprio, state=state, gripper=gripper)

    with LatentCache(path, mode="r") as c:
        assert c.trajectory_ids() == ["reach-v2/00000"]
        tr = c.read_trajectory("reach-v2/00000")
        assert tr["z"].shape == (T, 1, 4, 4, 8)
        assert tr["action"].shape == (T - 1, 4)
        assert tr["state"].shape == (T, 39)
        assert tr["proprio"].shape == (T, 4)
        np.testing.assert_allclose(tr["z"], z.numpy(), rtol=1e-5)
        np.testing.assert_allclose(tr["state"], state.numpy(), rtol=1e-5)


def test_write_and_read_regime(tmp_path):
    path = tmp_path / "mw__m.h5"
    T = 5
    with LatentCache(path, mode="w") as c:
        c.write_trajectory("t/0", z=torch.randn(T, 1, 2, 2, 4),
                           action=torch.randn(T - 1, 4), state=torch.randn(T, 39))
    ids = np.array([REGIME_TO_ID["free_space"], REGIME_TO_ID["contact_manipulation"],
                    REGIME_TO_ID["pre_grasp"], REGIME_TO_ID["gripper_actuation"]], dtype=np.int8)
    with LatentCache(path, mode="a") as c:
        c.write_regime("t/0", ids)
    with LatentCache(path, mode="r") as c:
        tr = c.read_trajectory("t/0")
        np.testing.assert_array_equal(tr["regime"], ids)
