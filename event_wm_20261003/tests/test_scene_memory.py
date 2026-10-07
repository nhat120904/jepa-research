"""CPU checks for the scene-memory model: FSQ round trip, exact persistence through a closed gate, causal export
shapes, and a few training steps on a synthetic cache (moving 'arm' blob + a block that teleports once)."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(ROOT))
from sm_model import FSQ, SceneMemory, run_episode  # noqa: E402


def synthetic_cache(path: Path, episodes=3, T=60, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.full((episodes * T, 64, 64, 3), 200, np.uint8)
    act = rng.normal(size=(episodes * T, 5)).astype(np.float32)
    term = np.zeros(episodes * T, bool)
    qpos = np.zeros((episodes * T, 35), np.float32)
    buttons = np.zeros((episodes * T, 2), np.int64)
    for e in range(episodes):
        bx, by = rng.integers(5, 55, size=2)
        for t in range(T):
            i = e * T + t
            if t == T // 2:
                bx, by = rng.integers(5, 55, size=2)
            obs[i, by:by + 4, bx:bx + 4] = (220, 30, 30)                     # 4 px block
            qpos[i, 14:17] = (bx / 64, by / 64, 0.02); qpos[i, :6] = np.sin(t / 5 + e)
            buttons[i] = (t >= T // 2, 0)
            ax, ay = int(32 + 20 * np.sin(t / 5 + e)), int(32 + 20 * np.cos(t / 7))
            obs[i, max(ay - 5, 0):ay + 5, max(ax - 5, 0):ax + 5] = (40, 40, 60)  # moving arm blob
        term[e * T + T - 1] = True
    for split in ("train", "val"):
        np.save(path / f"{split}_observations.npy", obs)
        np.save(path / f"{split}_actions.npy", act)
        np.save(path / f"{split}_terminals.npy", term)
        np.save(path / f"{split}_qpos.npy", qpos)
        np.save(path / f"{split}_button_states.npy", buttons)


def test_fsq_roundtrip():
    fsq = FSQ()
    idx = torch.randint(0, 5 ** 6, (100,))
    assert torch.equal(fsq.index(fsq.values(idx)), idx)
    v = fsq.values(idx)                                                     # (100, 6), channel last
    vm = v.T[None, :, :, None]                                              # (1, 6, 100, 1), channel first
    assert torch.equal(fsq.quantize(vm), vm)


def test_closed_gate_copies_code_exactly():
    torch.manual_seed(0)
    m = SceneMemory(width=32)
    with torch.no_grad():
        m.cell[-1].bias[-1] = -50.0                                         # gate pre-activation << 0
        f = torch.randn(2, 32, 16, 16)
        q0 = m.initial(f)
        q1, g, op = m.step(q0, torch.randn(2, 32, 16, 16))
    assert float(g.max()) == 0.0 and float(op.max()) == 0.0
    assert torch.equal(m.fsq.index(q0.permute(0, 2, 3, 1)), m.fsq.index(q1.permute(0, 2, 3, 1)))


def test_run_episode_shapes():
    m = SceneMemory(width=32).eval()
    obs = np.random.randint(0, 255, (20, 64, 64, 3), np.uint8)
    out = run_episode(m, obs, np.zeros((20, 5), np.float32), "cpu", chunk=7)
    assert out["codes"].shape == (20, 256) and out["opened"].shape == (20, 256)
    assert out["opened"][0].all()                                         # initial read


def test_training_steps():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        synthetic_cache(d)
        r = subprocess.run([sys.executable, str(ROOT / "sm_train.py"), "--cache", str(d), "--episodes", "3", "--val-episodes", "2",
                            "--steps", "3", "--batch", "4", "--clip", "16", "--width", "32", "--device", "cpu", "--out", str(d / "out")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        assert (d / "out" / "sm.pt").exists()
        for fam in ("cube", "puzzle"):
            r = subprocess.run([sys.executable, str(ROOT / "sm_diag.py"), "--model", str(d / "out" / "sm.pt"), "--cache", str(d),
                                "--family", fam, "--episodes", "2", "--val-episodes", "3", "--device", "cpu", "--out", str(d / f"diag_{fam}")],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-3000:]
            res = json.loads((d / f"diag_{fam}" / "diag.json").read_text())
            assert res["export_val"]["frames"] == 180 and "probes" in res
            assert np.load(d / f"diag_{fam}" / "export" / "train" / "codes.npy").shape == (120, 256)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
