"""Loader schema-adaptation tests using a MOCK dataset item — no upstream env,
no GPU, no data download. Verifies the parts that broke in the first draft:
tuple arity, visual rescaling to [0,255], action trim, gripper extraction."""

import torch

from data.loaders import _unpack_item, _build_transition, TransitionBatch


def _metaworld_item(T=5):
    obs = {"visual": torch.rand(T, 3, 8, 8), "proprio": torch.rand(T, 4)}  # visual in [0,1]
    act = torch.rand(T, 4)
    state = torch.rand(T, 39)
    return (obs, act, state, None, {})       # 5-tuple


def _droid_item(T=4):
    obs = {"visual": torch.rand(T, 3, 8, 8), "proprio": torch.rand(T, 7)}
    act = torch.rand(T, 7)
    state = torch.rand(T, 7)
    return (obs, act, state, torch.tensor(0.0))   # 4-tuple


def test_unpack_handles_4_and_5_tuples():
    o, a, s = _unpack_item(_metaworld_item())
    assert "visual" in o and a.shape[-1] == 4 and s.shape[-1] == 39
    o, a, s = _unpack_item(_droid_item())
    assert "visual" in o and a.shape[-1] == 7 and s.shape[-1] == 7


def test_build_transition_metaworld():
    obs, act, state = _unpack_item(_metaworld_item(T=5))
    tb = _build_transition(obs, act, state, env="metaworld", traj_id="reach-v2/0", task="reach-v2")
    assert isinstance(tb, TransitionBatch)
    assert tb.obs_visual.max() > 1.5            # rescaled to [0,255]
    assert tb.action.shape[0] == 5 - 1          # one fewer action than frames
    assert tb.state.shape == (5, 39)
    assert tb.gripper is not None and tb.gripper.shape == (5,)
    # gripper is proprio[:, 3] for metaworld
    torch.testing.assert_close(tb.gripper, tb.proprio[:, 3])


def test_build_transition_droid_gripper_idx6():
    obs, act, state = _unpack_item(_droid_item(T=4))
    tb = _build_transition(obs, act, state, env="droid", traj_id="droid/0", task="droid")
    assert tb.action.shape[0] == 4 - 1
    torch.testing.assert_close(tb.gripper, tb.proprio[:, 6])


def test_visual_already_255_not_double_scaled():
    obs = {"visual": torch.rand(3, 3, 8, 8) * 255.0, "proprio": torch.rand(3, 4)}
    tb = _build_transition(obs, torch.rand(3, 4), torch.rand(3, 39),
                           env="metaworld", traj_id="t/0", task="t")
    assert tb.obs_visual.max() <= 255.0 + 1e-3   # not rescaled again
