"""CPU checks for scene memory v2: memory rule, event merging, per-patch code locality, token dilation, and a smoke run of
all training stages + diagnostics on a synthetic cache (an 'arm' blob driven by the actions, a block that jumps once)."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(ROOT))
from sm2_model import (SceneCodes, SeeThrough, assemble_events, backdate, code_digits, dilate_tokens, memory_rule,  # noqa: E402
                       merge_onsets, see_codes, see_targets)


def test_memory_rule_waits_for_agent_and_debounces():
    T, N = 20, 2
    codes = np.zeros((T, N), np.int64)
    agent = np.zeros((T, N), bool)
    codes[8:, 0] = 5                       # token 0 changes at frame 8 while the agent covers it (frames 6-11)
    agent[6:12, 0] = True
    codes[15:17, 1] = 9                    # token 1 flickers for 2 frames (< k = 3)
    mem, changed = memory_rule(codes, agent, k=3)
    assert mem[2, 0] == 0 and not changed[:3].any()                      # first acceptance at frame 2 is not a change
    assert mem[13, 0] == 0 and mem[14, 0] == 5                            # agent leaves at 12, 3 frames with code 5 -> 14
    assert changed[:, 0].sum() == 1 and changed[14, 0]
    assert (mem[:, 1][mem[:, 1] >= 0] == 0).all() and not changed[:, 1].any()


def test_see_targets_only_for_unchanged_visits():
    T, N = 30, 2
    codes = np.zeros((T, N), np.int64)
    agent = np.zeros((T, N), bool)
    agent[5:10, :] = True                     # both tokens covered at 5-9
    codes[5:10, :] = 7                        # what the camera sees under the agent (tint)
    codes[10:, 1] = 4                         # token 1's scene changed during the visit
    mem, _, conf = memory_rule(codes, agent, 3, return_confirmed=True)
    y = see_targets(codes, agent, mem, conf)
    assert (y[5:10, 0] == 0).all()            # unchanged visit: the code before / after the visit
    assert (y[5:10, 1] == -1).all()           # changed visit: no target
    assert y[2, 0] == 0 and y[0, 0] == -1     # clean anchor once the memory holds the code


def test_assemble_events_attaches_late_nearby_changes():
    ch = np.zeros((100, 256), bool)
    ch[10, 5 * 16 + 5] = True
    ch[30, 5 * 16 + 6] = True                 # adjacent token, 20 frames later -> same interaction
    ch[35, 12 * 16 + 12] = True               # far away -> its own event
    ch[90, 5 * 16 + 5] = True                 # same place, much later -> new event
    ev = assemble_events(ch, np.array([0]), 100, gap=30)
    assert ev.tolist() == [[10, 30, 2], [35, 35, 1], [90, 90, 1]]


def test_backdate_to_fresh_onset():
    T = 40
    fresh = np.zeros((T, 2), np.int64); fresh[12:, 0] = 5          # see-through memory shows the new value from 12
    fresh[15:18, 1] = 9                                             # token 1: a transient see-through error
    fresh[3, 0] = 5                                                 # an early see-through error showing the new value
    conf = np.zeros((T, 2), np.int64); conf[20:, 0] = 5             # confirmed after the agent left, at 20
    ok = np.zeros((T, 2), bool); ok[:8, 0] = True; ok[20:, 0] = True  # old value confirmed agent-free until 7
    ch = np.zeros((T, 2), bool); ch[20, 0] = True
    out = backdate(ch, conf, ok, fresh, np.array([0]), T)
    assert out[:, 0].nonzero()[0].tolist() == [12] and not out[:, 1].any()


def test_see_codes_digit_order():
    torch.manual_seed(0)
    m = SeeThrough(3, 8, w=32).eval()
    c, p = see_codes(m, torch.rand(2, 3, 64, 64))
    assert c.shape == (2, 256) and ((p > 0) & (p <= 1)).all()
    d = code_digits(c, 3, 8)
    assert (d[..., 0] + 8 * d[..., 1] + 64 * d[..., 2] == c).all()


def test_merge_onsets_within_episode_only():
    on = np.zeros(40, bool)
    on[[3, 6, 12, 19, 20, 21]] = True
    ev = merge_onsets(on, np.array([0, 20]), 20, merge=5)
    assert ev.tolist() == [[3, 6], [12, 12], [19, 19], [20, 21]]


def test_scene_codes_are_patch_local():
    torch.manual_seed(0)
    m = SceneCodes(width=32).eval()
    x = torch.rand(1, 3, 64, 64)
    y = x.clone(); y[:, :, 20:24, 36:40] = torch.rand(1, 3, 4, 4)        # token (5, 9) only
    with torch.no_grad():
        cx, cy = m.codes(x)[0], m.codes(y)[0]
    diff = torch.nonzero(cx != cy).flatten().tolist()
    assert set(diff) <= {5 * 16 + 9}


def test_dilate_tokens():
    m = np.zeros((1, 256), bool); m[0, 5 * 16 + 9] = True
    d = dilate_tokens(m, 1).reshape(16, 16)
    assert d.sum() == 9 and d[4:7, 8:11].all()
    c = np.zeros((1, 256), bool); c[0, 0] = True
    assert dilate_tokens(c, 1).sum() == 4                                 # corner: no wrap-around


def synthetic_cache(path: Path, episodes=3, T=60, seed=0):
    rng = np.random.default_rng(seed)
    N = episodes * T
    obs = np.full((N, 64, 64, 3), 200, np.uint8)
    act = rng.uniform(-1, 1, size=(N, 5)).astype(np.float32)
    term = np.zeros(N, bool)
    qpos = np.zeros((N, 35), np.float32)
    buttons = np.zeros((N, 2), np.int64)
    for e in range(episodes):
        ax, ay = 32.0, 32.0
        bx, by = rng.integers(5, 55, size=2)
        for t in range(T):
            i = e * T + t
            if t == T // 2:
                bx, by = rng.integers(5, 55, size=2)
            obs[i, by:by + 4, bx:bx + 4] = (220, 30, 30)
            qpos[i, 14:17] = (bx / 64, by / 64, 0.02); qpos[i, :2] = (ax, ay)
            buttons[i] = (t >= T // 2, 0)
            obs[i, int(ay) - 5:int(ay) + 5, int(ax) - 5:int(ax) + 5] = (40, 40, 60)
            ax, ay = np.clip(ax + 3 * act[i, 0], 6, 58), np.clip(ay + 3 * act[i, 1], 6, 58)   # the 'arm' follows the action
        term[e * T + T - 1] = True
    for split in ("train", "val"):
        np.save(path / f"{split}_observations.npy", obs)
        np.save(path / f"{split}_actions.npy", act)
        np.save(path / f"{split}_terminals.npy", term)
        np.save(path / f"{split}_qpos.npy", qpos)
        np.save(path / f"{split}_button_states.npy", buttons)


def test_pipeline_smoke():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        synthetic_cache(d)
        out = d / "run"
        small = ["--cache", str(d), "--out", str(out), "--episodes", "3", "--val-episodes", "3", "--device", "cpu", "--batch", "8",
                 "--agent-width", "8", "--agent-steps", "3", "--label-fit-pairs", "64", "--chunk", "32", "--seg-width", "16",
                 "--seg-steps", "3", "--scene-width", "32", "--scene-steps", "3", "--clips", "2", "--clip", "4"]
        for stage in ("agent", "label", "seg", "segpred", "scene", "stargets", "seethru:learned", "seethru:pixel"):
            st, _, sp = stage.partition(":")
            extra = ["--space", sp, "--see-width", "16", "--see-steps", "3"] if sp else []
            r = subprocess.run([sys.executable, str(ROOT / "sm2_train.py"), "--stage", st] + small + extra, capture_output=True, text=True)
            assert r.returncode == 0, (stage, r.stderr[-3000:])
        assert np.load(out / "teacher_val.npy").shape == (180, 32) and np.load(out / "seg_val.npy").shape == (180, 256)
        for fam in ("scene", "puzzle"):
            r = subprocess.run([sys.executable, str(ROOT / "sm2_diag.py"), "--run", str(out), "--cache", str(d), "--family", fam,
                                "--val-episodes", "3", "--probe-steps", "5", "--fit-episodes", "2", "--device", "cpu", "--out", str(d / f"diag_{fam}")],
                               capture_output=True, text=True)
            assert r.returncode == 0, (fam, r.stderr[-3000:])
            res = json.loads((d / f"diag_{fam}" / "diag.json").read_text())
            assert res["frames"] == 180 and all("criteria" in res[v] for v in ("learned", "pixel_tokens", "see_learned", "see_pixel"))
            assert np.load(d / f"diag_{fam}" / "export" / "val" / "see_learned" / "mem.npy").shape == (180, 256)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
