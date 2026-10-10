"""CPU checks for method/: the ported rules equal the frozen originals where they should, shapes and the new pieces."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "method"
sys.path.insert(0, str(ROOT))
from events import app_thresholds, rest_runs_vec, transition  # noqa: E402
from memory_entities import digits, token_centres  # noqa: E402
from world_model import Model, finite_rest_support, make_h, make_wm, prototypes  # noqa: E402


def rest_runs_ref(pos, app, vis, a, b, r_pos, r_app, m):
    """docs/generic_state_20261007/base_source/u_events.rest_runs, verbatim."""
    runs, cur, cand = [], None, None

    def near(run, p, c):
        return np.hypot(*(p - run[2] / run[4])) <= r_pos and np.abs(c - run[3] / run[4]).max() <= r_app

    for t in range(a, b):
        if not vis[t]:
            continue
        p, c = pos[t], app[t]
        if cur is not None and near(cur, p, c):
            cur[1] = t; cur[2] = cur[2] + p; cur[3] = cur[3] + c; cur[4] += 1; cand = None
            continue
        if cand is not None and near(cand, p, c):
            cand[1] = t; cand[2] = cand[2] + p; cand[3] = cand[3] + c; cand[4] += 1
            if cand[4] >= m:
                if cur is not None:
                    runs.append(cur)
                cur, cand = cand, None
        else:
            cand = [t, t, p.astype(np.float64).copy(), c.astype(np.float64).copy(), 1]
    if cur is not None:
        runs.append(cur)
    return runs


def transition_ref(pos, app, r0, r1, r_pos, r_app):
    """u_events.transition, verbatim."""
    s = np.arange(r0[1], r1[0] + 1)

    def at(r):
        return (np.linalg.norm(pos[s] - r[2] / r[4], axis=-1) <= r_pos) & (np.abs(app[s] - r[3] / r[4]).max(-1) <= r_app)

    stay = np.flip(np.cumprod(np.flip(at(r1)))).astype(bool)
    arr = int(s[stay.argmax()]) if stay.any() else int(r1[0])
    pre = s[(s < arr) & at(r0)]
    dep = int(pre.max()) if len(pre) else int(r0[1])
    return dep + 1, arr


def test_rest_runs_and_transition_match_u_events():
    rng = np.random.default_rng(0)
    T, K, A = 400, 5, 6
    app = np.zeros((T, K, A))
    for k in range(K):                                          # piecewise-constant codes with flicker and occlusion
        level = rng.integers(0, 5, A)
        for t in range(T):
            if rng.random() < 0.02:
                level = rng.integers(0, 5, A)
            app[t, k] = level / 4.0
        app[rng.random(T) < 0.03, k] = rng.integers(0, 5, (int((rng.random(T) < 0.03).sum()) or 1, A))[0] / 4.0
    ok = rng.random((T, K)) > 0.2
    pos = np.zeros((T, 2))
    runs = rest_runs_vec(app, ok, 0, 0.0625, 5)
    for k in range(K):
        ref = rest_runs_ref(pos, app[:, k], ok[:, k], 0, T, 1.0, 0.0625, 5)
        assert [(r[0], r[1], r[4]) for r in ref] == [(r[0], r[1], r[3]) for r in runs[k]]
        for i in range(1, len(ref)):
            a0 = transition_ref(pos, app[:, k], ref[i - 1], ref[i], 1.0, 0.0625)
            r0 = [runs[k][i - 1][0], runs[k][i - 1][1], runs[k][i - 1][2], runs[k][i - 1][3]]
            r1 = [runs[k][i][0], runs[k][i][1], runs[k][i][2], runs[k][i][3]]
            assert transition(app[:, k], r0, r1, 0.0625) == a0


def test_quantized_threshold_rule():
    n = 5000
    ends = np.array([999, 1999, 2999, 3999, 4999]); starts = np.r_[0, ends[:-1] + 1]
    app = np.zeros((n, 3, 6), np.uint8); app[2500:, 1, 2] = 1
    z = {"n": n, "area": np.ones((n, 3), np.uint8), "free": np.ones((n, 3), bool), "app": app}
    r_app, thr_app, rep = app_thresholds(z, starts, ends, 0.25, np.random.default_rng(0))
    assert rep["rule"] == "quantized" and r_app == 0.0625 and thr_app == 0.125


def test_digits_follow_see_codes_order():
    nl, nd = 5, 6
    arg = np.array([[1, 0, 4, 2, 3, 0]])
    code = (arg * (nl ** np.arange(nd))).sum(-1)                  # frontend.see_codes: sum arg_j * nl^j
    assert (digits(code) == arg).all()
    c = token_centres()
    assert tuple(c[17]) == (5.5, 5.5) and tuple(c[0]) == (1.5, 1.5)


def toy_model(K=4, A=6):
    D = A + 3
    ck = {"K": K, "D": D, "thr_pos": 4.0, "tol_pos": 2.0, "thr_app": 0.125, "thr_app_id": np.full(K, 0.125),
          "wm": make_wm(K, D).state_dict(), "h": make_h(K, D, 32, True).state_dict(), "h_width": 32, "h_absdiff": True,
          "proto": [np.c_[np.full((2, 2), 8.0), np.array([[0.0] * A, [0.25] * A]), np.zeros(2)].astype(np.float32) for _ in range(K)],
          "occ_min": 0.0, "cover_rule": False, "event_pos_only": True, "max_candidates": 0}
    return Model(ck, "cpu")


def test_world_model_shapes_and_unknown_goal():
    M = toy_model()
    S = np.zeros((4, 9), np.float32); S[:, :2] = 8.0
    G = S.copy(); G[2, 2:-1] = 0.25
    known = np.array([True, True, False, True])
    assert M.at_goal(S, G, known) and not M.at_goal(S, G)        # entity 2 differs but its goal is unknown
    cands = M.candidates(S, G, known)
    assert all(e != 2 or np.allclose(x[2:-1], 0.25) for e, x in cands)   # entity 2: only its prototypes
    nxt = M.step(S, cands[:3])
    assert nxt.shape == (3, 4, 9)
    hv = M.heuristic(np.stack([S, S]), G, known)
    assert hv.shape == (2,)


def test_prototypes_are_frequent_codes_not_means():
    rests = {"entity": np.array([0, 0, 0]), "start": np.array([0, 100, 300]), "end": np.array([89, 289, 309]),
             "app": np.array([[0.0] * 6, [1.0] * 6, [0.5] * 6], np.float32), "pos": np.array([[8.0, 8.0]], np.float32)}
    p = prototypes(rests, 1, 6, True, 4.0, 0.125, 2, np.random.default_rng(0))[0]
    assert p.shape == (2, 9) and np.allclose(p[0, 2:-1], 1.0) and np.allclose(p[1, 2:-1], 0.0)


def test_finite_support_keeps_continuous():
    S = np.zeros((50, 1, 3), np.float32); S[:, 0, 0] = np.linspace(0, 1, 50); S[:, 0, 1] = np.tile([0.0, 0.25], 25)
    f = finite_rest_support(S)
    assert f[0][0] is None and f[0][1] == [0.0, 0.25]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)


# ---------------- object track ----------------
from objects import place_tokens, read_frame, reader_context, snap_reading, violations  # noqa: E402
import events_objects  # noqa: E402,F401  (imports cleanly)


def test_violations_count_frames_with_one_label_twice():
    F = np.array([0, 0, 1, 1, 2])
    assert violations(F, np.array([0, 0, 0, 1, 0])) == 1          # frame 0 holds label 0 twice
    assert violations(F, np.array([0, 1, 0, 1, 0])) == 0


def test_snap_reading_two_sides():
    mu, U = np.zeros(3), np.array([1.0, 0.0, 0.0])
    d = {"snap": (mu, U, 0.0, np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]), np.array([10.0, 20.0]), np.array([11.5, 20.5]))}
    a1, p1 = snap_reading(np.array([0.3, 5, 5]), np.array([37.5, 24.5]), d)
    a0, p0 = snap_reading(np.array([-0.2, 5, 5]), np.array([39.0, 25.0]), d)
    assert a1[0] == 1.0 and a0[0] == -1.0
    assert np.allclose(p1, [11.5, 20.5]) and np.allclose(p0, [10.0, 20.0])         # the position snaps with the state
    a, p = snap_reading(np.array([0.3, 5, 5]), np.array([1.0, 2.0]), {})
    assert (a == np.array([0.3, 5, 5])).all() and (p == np.array([1.0, 2.0])).all()  # not a two-state place: unchanged
    d5 = {"snap": d["snap"][:5]}                                                   # tables built before positions snapped
    assert np.allclose(snap_reading(np.array([0.3, 5, 5]), np.array([1.0, 2.0]), d5)[1], [1.0, 2.0])


def test_place_tokens_overlap():
    disc = np.zeros((64, 64), bool); disc[5:7, 9:11] = True      # rows 5-6, columns 9-10 -> token (1, 2) only
    idents = [{"anchor": "location", "disc": disc}, {"anchor": "mover"}]
    assert place_tokens(idents, np.array([18, 19, 40]))[0] == [0]   # token 18 = row 1, column 2


def test_read_frame_finds_a_mover_on_the_typical_background():
    L = 8
    trgb = np.full((64, 64, 3), [40, 50, 70], np.float32)
    from objects import quant_key
    x = np.broadcast_to(trgb, (64, 64, 3)).astype(np.uint8).copy()
    tkey = quant_key(x, L)
    x[20:25, 30:35] = [200, 40, 40]                                 # a red 5 x 5 object
    P = {"L": L, "tkey": tkey, "trgb": trgb, "tau_c": 0.03, "agent_bin": np.zeros(L ** 3, bool)}
    red = np.array([200, 40, 40], np.float32); proto = (red / (red.sum() + 1.0))[:2]
    idents = [{"anchor": "mover", "proto": proto.astype(np.float32), "colour": np.r_[proto, 1 - proto.sum()].astype(np.float32), "area": 25.0}]
    pos, app, area, ag_r = read_frame(x, np.zeros((64, 64), bool), P, idents, np.zeros((64, 64), bool), reader_context(idents, P))
    assert area[0] == 25 and np.allclose(pos[0], [32.0, 22.0])     # (u = column, v = row)
    ag = np.zeros((64, 64), bool); ag[20:25, 30:33] = True          # the mask covers 15 of its 25 pixels
    _, _, area2, ag_r = read_frame(x, ag, P, idents, np.zeros((64, 64), bool), reader_context(idents, P))
    assert area2[0] == 25 and not ag_r.any()                        # red is no agent colour: seen through the mask
    P2 = dict(P, agent_bin=np.zeros(L ** 3, bool)); P2["agent_bin"][quant_key(x[20:21, 30:31], L)[0, 0]] = True
    _, _, area3, _ = read_frame(x, ag, P2, idents, np.zeros((64, 64), bool), reader_context(idents, P2))
    assert area3[0] == 0                                            # an agent colour: hidden, 10 of 25 visible -> unobserved


def test_stab_groups_assign_a_long_window_to_the_moment_nearest_its_centre():
    from events_objects import stab_groups
    # press A at ~10: lights 0, 1 in [5, 12]; press B at ~40: light 2 in [36, 44]; light 3 hidden over both, [11, 50]
    ch = [(5, 12, 0, 0), (6, 12, 1, 0), (36, 44, 2, 0), (11, 50, 3, 0)]
    g = stab_groups(ch, 0)
    assert [sorted(c[2] for c in grp) for grp in g] == [[0, 1], [2, 3]]   # first-moment rule: [[0, 1, 3], [2]]
    assert len(stab_groups([(5, 12, 0, 0), (13, 20, 1, 0)], 0)) == 2         # disjoint windows: two events
    assert len(stab_groups([(5, 12, 0, 0), (13, 20, 1, 0)], 2)) == 1         # a slack of 2 joins them


def test_stab_groups_terminates_on_a_change_met_within_one_frame():
    from events_objects import stab_groups
    g = stab_groups([(11, 10, 0, 0), (10, 12, 1, 0), (30, 31, 2, 0)], 0)    # transition() can return t0 = t1 + 1
    assert [sorted(c[2] for c in grp) for grp in g] == [[0, 1], [2]]


def test_skill_architectures_output_action_chunks():
    import torch
    from skill import make_skill
    px = torch.rand(2, 6, 64, 64); e = torch.tensor([0, 1]); cur = torch.rand(2, 6) * 60; tgt = torch.rand(2, 6) * 60
    for arch in ("cnn", "keypoints"):
        out = make_skill(4, 3, chunk=8, cond="spatial", arch=arch)(px, e, cur, tgt, torch.zeros(2, 2))
        assert out.shape == (2, 8, 5)
