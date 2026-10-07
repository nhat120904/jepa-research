"""OGBench reference agent (impls/main.py, unmodified) with an optional input transform -- fairness control.

--input_mode raw     the published protocol: the env's state observation vector, as in the OGBench paper.
--input_mode entity  the same vector plus the features our state adapter (s_entities.py) derives from it: the entity
                     table (per entity u, v, a0, a1, a2, covered) and the end-effector point, standardised with TRAIN
                     statistics; constant columns dropped. The same transform is applied to dataset observations, the
                     dataset goals (sampled from dataset observations) and the evaluation observations / goals.
Everything else (agent code, hyperparameters, dataset, training steps, evaluation) is OGBench's own main.py.
"""

import json
import os
import sys

IMPLS = os.environ["OGBENCH_IMPLS"]
sys.path.insert(0, IMPLS)
sys.path.insert(0, os.environ["EVENT_WM_SOURCE"])

import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402
from absl import app, flags  # noqa: E402
from gymnasium.spaces import Box  # noqa: E402

import utils.env_utils as env_utils  # noqa: E402
import utils.log_utils as log_utils  # noqa: E402
from utils.datasets import Dataset  # noqa: E402

flags.DEFINE_string("input_mode", "raw", "raw | entity")
flags.DEFINE_string("layout", None, "layout.json from s_entities.py (entity mode)")
flags.DEFINE_string("dataset_dir", None, "directory with <dataset>.npz and <dataset>-val.npz")
FLAGS = flags.FLAGS


class EntityFeatures:
    def __init__(self, layout_path, train_obs):
        import types

        # s_entities imports sfa_code only for a numpy helper used by its own main(); sfa_code pulls in the torch
        # pipeline, which this JAX venv does not have. state_entities itself is pure numpy.
        sys.modules.setdefault("sfa_code", types.SimpleNamespace(two_means_threshold=None))
        from s_entities import state_entities

        self.L = json.loads(open(layout_path).read())
        self.state_entities = state_entities
        f = self.raw(train_obs)
        sd = f.std(0)
        self.keep = sd > 1e-6
        self.mu, self.sd = f.mean(0)[self.keep], sd[self.keep]

    def raw(self, obs):
        out = []
        for i in range(0, len(obs), 200000):
            S, eff = self.state_entities(np.asarray(obs[i:i + 200000], np.float32), self.L)
            out.append(np.concatenate([S.reshape(len(S), -1), eff], -1))
        return np.concatenate(out).astype(np.float32)

    def __call__(self, obs):
        obs = np.asarray(obs, np.float32)
        one = obs.ndim == 1
        o = obs[None] if one else obs
        z = (self.raw(o)[:, self.keep] - self.mu) / self.sd
        y = np.concatenate([o, z.astype(np.float32)], -1)
        return y[0] if one else y


class TransformObs(gymnasium.Wrapper):
    def __init__(self, env, fn, dim):
        super().__init__(env)
        self.fn = fn
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def reset(self, **kw):
        ob, info = self.env.reset(**kw)
        if "goal" in info:
            info["goal"] = self.fn(info["goal"])
        return self.fn(ob), info

    def step(self, action):
        ob, r, term, trunc, info = self.env.step(action)
        return self.fn(ob), r, term, trunc, info


def make_env_and_datasets(dataset_name, frame_stack=None):
    assert frame_stack is None
    env, train, val = ogbench.make_env_and_datasets(dataset_name, dataset_dir=FLAGS.dataset_dir, compact_dataset=True)
    report = {"input_mode": FLAGS.input_mode, "dataset": dataset_name, "train_obs": list(train["observations"].shape),
              "val_obs": list(val["observations"].shape)}
    if FLAGS.input_mode == "entity":
        fe = EntityFeatures(FLAGS.layout, train["observations"])
        train = dict(train); val = dict(val)
        train["observations"] = fe(train["observations"]); val["observations"] = fe(val["observations"])
        env = TransformObs(env, fe, train["observations"].shape[1])
        report.update(entity_columns_kept=int(fe.keep.sum()), obs_dim=int(train["observations"].shape[1]), layout=FLAGS.layout)
    elif FLAGS.input_mode != "raw":
        raise ValueError(FLAGS.input_mode)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, "input_transform.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("INPUT_TRANSFORM", json.dumps(report), flush=True)
    env.reset()
    return env, Dataset.create(**train), Dataset.create(**val)


_orig_wandb = log_utils.setup_wandb


def setup_wandb_offline(**kw):
    kw["mode"] = "offline"
    return _orig_wandb(**kw)


env_utils.make_env_and_datasets = make_env_and_datasets
log_utils.setup_wandb = setup_wandb_offline
import main as ogmain  # noqa: E402  (binds the patched functions at import)

assert ogmain.make_env_and_datasets is make_env_and_datasets

if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    ogmain.setup_wandb = setup_wandb_offline
    app.run(ogmain.main)
