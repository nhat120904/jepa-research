"""ti_wm.dinowm_scorer adapter: the kinematic PD agent path equals gym-pusht's; macro-action shapes; goal agent parsing."""
import unittest

import numpy as np

from ti_wm import dinowm_scorer as dw


class Adapter(unittest.TestCase):
    def test_macro_shape_and_hold(self):
        pos, vel = np.array([[100., 200.]]), np.zeros((1, 2))
        chunk = np.tile(np.array([[150., 220.]]), (1, 8, 1))
        a = dw.macro_actions(pos, vel, chunk)
        self.assertEqual(tuple(a.shape), (1, dw.MACRO, 10))

    def test_pd_matches_gym_pusht(self):
        try:
            import gymnasium as gym
            import gym_pusht  # noqa: F401
        except ImportError:
            self.skipTest("gym-pusht not importable here")
        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
        obs, _ = env.reset(seed=30250)
        u = env.unwrapped
        # the goal-agent parser reads the renderer's orientation correctly (image column = x, row = y)
        np.testing.assert_allclose(dw.agent_from_frame(obs["pixels"]), obs["agent_pos"], atol=6.0)
        self.assertEqual(dw.canvas224(u).shape, (224, 224, 3))
        rng = np.random.default_rng(0)
        pos0, vel0 = np.array(u.agent.position), np.array(u.agent.velocity)
        targets = np.clip(pos0 + rng.normal(0, 40, (8, 2)), 20, 490)
        live = []
        for tgt in targets:
            live.append(np.array(u.agent.position))
            env.step(tgt)
        pred = dw.pd_positions(pos0[None], vel0[None], targets[None])[0]
        err = np.abs(pred - np.stack(live)).max()
        env.close()
        self.assertLess(err, 1e-6, f"PD path differs from gym-pusht by {err}")

    def test_agent_from_frame(self):
        frame = np.full((96, 96, 3), 255, np.uint8)
        frame[40:44, 60:64] = dw.ROYAL_BLUE
        xy = dw.agent_from_frame(frame)
        np.testing.assert_allclose(xy, [62 * 512 / 96, 42 * 512 / 96], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
