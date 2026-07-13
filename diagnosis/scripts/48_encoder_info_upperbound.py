"""Encoder information upper-bound probes — WHAT does the frozen dino_wm_metaworld
latent actually carry of the true-state oracle cost?

Reviewer-1 gap (paper sec:ladder + sec:e1): the oracle ladder shows every
frozen-encoder cost fails under PERFECT dynamics while the true-state cost solves
push 16/16. Two readings survive: (a) the cost/optimizer interface is EXPLOITABLE,
or (b) the frozen representation LACKS the information contact planning needs. This
script upper-bounds (b): it trains small readouts from the frozen latent for every
quantity the winning state-oracle cost consumed, on a held-out trajectory split,
and reports how precisely each is decodable.

The state-oracle cost (scripts/29 roll_cost, the push-16/16 winner) is
    c*(s) = ‖obj − g‖ + 0.5·‖hand − obj‖ .
So the decision-relevant quantities are: object xyz (already measured — scripts/21,
92% <5cm), end-effector xyz (scripts/19), and the MISSING pieces this script adds:

  a. gripper aperture  state[3]                 — scalar  (R² + MAE)
  b. hand−object relative vector  ee − obj (3d) — the cost's shaping term directly
  c. the full cost value c*(s) from [z_t, z_g]  — the single most decision-relevant
        scalar: if it decodes precisely, the latent contains everything the winning
        oracle cost used (z_g = the trajectory's terminal/goal latent, exactly what
        the oracle encodes as z_goal).
  d. contact indicator  ‖Δobj‖>5mm at t→t+1     — AUROC from z_t ALONE (may be
        unknowable) and from the PAIR (z_t,z_{t+1}) (knowable iff motion is visible).

Everything is a small MLP/transformer readout on the CACHED latents; encoder and
predictor stay frozen. CPU-only for the on-policy pass (no model load). The
optional off-policy pass (--offpolicy, needs a GPU: encoder + EGL render) rolls
RANDOM actions and re-evaluates the single-frame probes (gripper, ee−obj, object)
on the distribution the planner actually scores — the on/off gap that scripts/34
found collapses the object readout.

    # on-policy (CPU, no GPU):
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/48_encoder_info_upperbound.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld

    # + off-policy single-frame probes (GPU: encoder + render):
    .venv/bin/python scripts/48_encoder_info_upperbound.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld --offpolicy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from data.latent_cache import ID_TO_REGIME  # noqa: E402
from models.heads.mixture_predictor import flatten_tokens  # noqa: E402
from stratification.metaworld_regimes import (  # noqa: E402
    EE_SLICE, GRIPPER_IDX, OBJECT_SLICE, OBJECT_MOVE_THRESHOLD,
)

PUSH_RADIUS, PICK_RADIUS = 0.05, 0.07
W_HAND = 0.5  # scripts/29 roll_cost hand-approach weight


# --------------------------------------------------------------------------- #
# Probe heads (mean+max pooled MLPs — same family as models/probes/ObjectProbe)
# --------------------------------------------------------------------------- #
class PooledMLP(nn.Module):
    """Mean+max-pooled token MLP over ONE latent frame -> out_dim."""

    def __init__(self, latent_dim: int, out_dim: int, hidden: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def _pool(self, z):
        tok = flatten_tokens(z, self.latent_dim)
        return torch.cat([tok.mean(1), tok.max(1).values], -1)

    def forward(self, z):
        return self.net(self._pool(z))


class PairPooledMLP(PooledMLP):
    """Mean+max pool of TWO latent frames [z_t ‖ z_g] -> out_dim (4*D input)."""

    def __init__(self, latent_dim: int, out_dim: int, hidden: int = 512):
        nn.Module.__init__(self)
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(4 * latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, z_a, z_b):
        return self.net(torch.cat([self._pool(z_a), self._pool(z_b)], -1))


# --------------------------------------------------------------------------- #
# Data: preload the strided transitions (login node has TBs of RAM)
# --------------------------------------------------------------------------- #
def load_dataset(cache_path, regime_by_traj, step, max_traj=0):
    """Return per-transition CPU tensors + terminal (goal) latent/object per traj.

    z_t, z_t1: (N,V,H,W,D); state_t/state_t1: (N,39); z_g: (N,V,H,W,D) terminal
    latent of the transition's trajectory; tid: (N,) trajectory id (for the split).
    """
    z_t, z_t1, z_g, st_t, st_t1, goal_obj, tids, tasks = [], [], [], [], [], [], [], []
    with LatentCache(cache_path, mode="r") as cache:
        all_ids = cache.trajectory_ids()
        if max_traj:
            all_ids = all_ids[:max_traj]
        for ti, tid in enumerate(all_ids):
            traj = cache.read_trajectory(tid)
            z = np.asarray(traj["z"], dtype=np.float32)      # (T,V,H,W,D)
            st = np.asarray(traj["state"], dtype=np.float32)  # (T,39)
            T = int(traj["action"].shape[0])
            n = T // step
            if n == 0:
                continue
            z_term = z[-1]                                    # goal latent (expert terminal)
            obj_term = st[-1, OBJECT_SLICE]                   # goal object xyz
            for k in range(n):
                i0 = k * step
                z_t.append(z[i0]); z_t1.append(z[i0 + step]); z_g.append(z_term)
                st_t.append(st[i0]); st_t1.append(st[i0 + step]); goal_obj.append(obj_term)
                tids.append(tid); tasks.append(tid.split("/")[0])
            if (ti + 1) % 120 == 0:
                print(f"  loaded {ti+1}/{len(all_ids)} trajectories", flush=True)
    z_t = torch.from_numpy(np.stack(z_t)); z_t1 = torch.from_numpy(np.stack(z_t1))
    z_g = torch.from_numpy(np.stack(z_g))
    st_t = torch.from_numpy(np.stack(st_t)); st_t1 = torch.from_numpy(np.stack(st_t1))
    goal_obj = torch.from_numpy(np.stack(goal_obj))
    return dict(z_t=z_t, z_t1=z_t1, z_g=z_g, st_t=st_t, st_t1=st_t1, goal_obj=goal_obj,
                tid=np.asarray(tids), task=np.asarray(tasks))


def targets_from_states(st_t, st_t1):
    """Compute the four decode targets from raw 39-dim states (+goal per traj)."""
    ee = st_t[:, EE_SLICE]
    obj = st_t[:, OBJECT_SLICE]
    grip = st_t[:, GRIPPER_IDX:GRIPPER_IDX + 1]
    rel = ee - obj                                            # hand - object (3d)
    dobj = (st_t1[:, OBJECT_SLICE] - obj).norm(dim=-1)
    contact = (dobj > OBJECT_MOVE_THRESHOLD).float()          # (N,)
    return dict(grip=grip, rel=rel, contact=contact, ee=ee, obj=obj)


def cost_target(st_t, z_g_obj):
    """c*(s) = ‖obj − g‖ + 0.5·‖hand − obj‖ ; g = object at the goal (terminal) frame."""
    ee = st_t[:, EE_SLICE]; obj = st_t[:, OBJECT_SLICE]
    d_obj = (obj - z_g_obj).norm(dim=-1)
    d_app = (ee - obj).norm(dim=-1)
    return d_obj + W_HAND * d_app


def split_by_trajectory(tids, val_frac, seed):
    uniq = sorted(set(tids.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    val = set(uniq[: max(1, int(len(uniq) * val_frac))])
    val_mask = np.array([t in val for t in tids])
    return ~val_mask, val_mask


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def vec_metrics(err_m):
    """err_m: (N,) L2 error in metres -> dict of cm/% metrics (object convention)."""
    return dict(median_cm=100 * float(np.median(err_m)),
                mean_cm=100 * float(err_m.mean()),
                p90_cm=100 * float(np.percentile(err_m, 90)),
                frac_within_5cm=float((err_m < PUSH_RADIUS).mean()),
                frac_within_7cm=float((err_m < PICK_RADIUS).mean()))


def scalar_metrics(pred, true):
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    ae = np.abs(pred - true)
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum()) + 1e-12
    return dict(r2=1.0 - ss_res / ss_tot, mae=float(ae.mean()),
                median_ae=float(np.median(ae)), true_std=float(true.std()),
                nmae=float(ae.mean() / (true.std() + 1e-12)))


# --------------------------------------------------------------------------- #
def train_head(head, forward_fn, target, train_idx, val_idx, *, epochs, bs, lr,
               device, rng, bce=False, name=""):
    """Generic minibatch trainer. forward_fn(idx_tensor)->pred; target aligned to
    the full record index space."""
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if bce else nn.MSELoss()
    tgt = target.to(device)
    tr = train_idx.copy()
    for ep in range(epochs):
        rng.shuffle(tr)
        head.train(); tot = 0.0; nb = 0
        for lo in range(0, len(tr), bs):
            idx = torch.as_tensor(tr[lo:lo + bs], dtype=torch.long)
            pred = forward_fn(idx)
            y = tgt[idx]
            if bce:
                pred = pred.reshape(-1); y = y.reshape(-1)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        # quick val loss
        head.eval()
        with torch.no_grad():
            vi = torch.as_tensor(val_idx, dtype=torch.long)
            vp = forward_fn(vi); vy = tgt[vi]
            if bce:
                vloss = loss_fn(vp.reshape(-1), vy.reshape(-1))
            else:
                vloss = loss_fn(vp, vy)
        print(f"    [{name}] epoch {ep+1}/{epochs}  train={tot/max(nb,1):.5f}  val={float(vloss):.5f}",
              flush=True)
    head.eval()
    return head


def batched_forward(head, get_inputs, n, device, bs=4096):
    """Run head over indices [0..n) in chunks; get_inputs(idx)->tuple of tensors."""
    outs = []
    with torch.no_grad():
        for lo in range(0, n, bs):
            idx = torch.arange(lo, min(lo + bs, n))
            ins = get_inputs(idx)
            outs.append(head(*ins).cpu())
    return torch.cat(outs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="dino_wm_metaworld")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--max-traj", type=int, default=0, help="limit #trajectories (smoke test)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offpolicy", action="store_true",
                    help="also eval single-frame probes off-policy (needs GPU: encoder+render)")
    ap.add_argument("--offpolicy-only", action="store_true",
                    help="skip training: load --probes-ckpt and run ONLY the off-policy eval")
    ap.add_argument("--probes-ckpt", default=None,
                    help="pre-trained probe bundle (from a prior on-policy run) for --offpolicy-only")
    ap.add_argument("--op-episodes", type=int, default=10)
    ap.add_argument("--op-max-steps", type=int, default=60)
    ap.add_argument("--op-collect-every", type=int, default=4)
    ap.add_argument("--op-tasks", nargs="+", default=["mw-push", "mw-pick-place", "mw-reach"])
    ap.add_argument("--out-csv", default="results/encoder_info_upperbound.csv")
    ap.add_argument("--out-md", default="results/encoder_info_upperbound.md")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "16")))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if (args.offpolicy and torch.cuda.is_available()) else "cpu")
    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, cfg["dataset"]["name"])
    regime_by_traj = read_regimes(cache_path)

    # ---- off-policy-only: load trained probes, eval on random-rollout latents --- #
    if args.offpolicy_only:
        ckpt = torch.load(args.probes_ckpt, map_location=device, weights_only=False)
        ld, hid = ckpt["latent_dim"], ckpt["hidden"]
        head_g = PooledMLP(ld, 1, hid).to(device); head_g.load_state_dict(ckpt["gripper"]); head_g.eval()
        head_rel = PooledMLP(ld, 3, hid).to(device); head_rel.load_state_dict(ckpt["rel"]); head_rel.eval()
        off = run_offpolicy(args, cfg, device, head_g, head_rel, ld)
        import pandas as pd
        rows = []
        for tgt_name, o in off.items():
            if "r2" in o:
                rows.append(dict(target=tgt_name, offp=f"R²={o['r2']:.3f} MAE={o['mae']:.3f} nMAE={o['nmae']:.2f}", **o))
            else:
                rows.append(dict(target=tgt_name, offp=f"median={o['median_cm']:.2f}cm <5cm={o['frac_within_5cm']*100:.1f}%", **o))
        outp = args.out_csv.replace(".csv", "_offpolicy.csv")
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(outp, index=False)
        print(f"\nwrote {outp}", flush=True)
        return 0

    print(f"loading cache {cache_path} (step={args.step}) ...", flush=True)
    D = load_dataset(cache_path, regime_by_traj, args.step, args.max_traj)
    N = D["z_t"].shape[0]
    latent_dim = int(D["z_t"].shape[-1])
    print(f"transitions: {N}  latent_dim={latent_dim}", flush=True)

    tgt = targets_from_states(D["st_t"], D["st_t1"])
    # goal object g = object at the trajectory's TERMINAL (goal) frame — matches the
    # oracle's goal_obj = probe(z_goal), z_goal = encoded expert-terminal frame.
    c_true = cost_target(D["st_t"], D["goal_obj"])

    train_mask, val_mask = split_by_trajectory(D["tid"], args.val_frac, args.seed)
    train_idx = np.nonzero(train_mask)[0]; val_idx = np.nonzero(val_mask)[0]
    print(f"split: train={len(train_idx)} val={len(val_idx)} trajectories", flush=True)

    z_t = D["z_t"].to(device); z_g = D["z_g"].to(device)

    rows = []           # CSV rows
    off_rows = {}       # target -> off-policy metrics

    # ---- a. gripper aperture (scalar) -------------------------------------- #
    head_g = PooledMLP(latent_dim, 1, args.hidden).to(device)
    train_head(head_g, lambda i: head_g(z_t[i]).reshape(-1), tgt["grip"].reshape(-1),
               train_idx, val_idx, epochs=args.epochs, bs=args.bs, lr=args.lr,
               device=device, rng=rng, name="gripper")
    gp = batched_forward(head_g, lambda i: (z_t[i],), N, device).reshape(-1).numpy()
    m = scalar_metrics(gp[val_idx], tgt["grip"].reshape(-1).numpy()[val_idx])
    rows.append(dict(target="gripper_aperture(state[3])", kind="scalar", n=len(val_idx),
                     onp_metric=f"R²={m['r2']:.3f} MAE={m['mae']:.3f} (std={m['true_std']:.3f}, nMAE={m['nmae']:.2f})",
                     **{f"onp_{k}": v for k, v in m.items()}))

    # ---- b. hand-object relative vector (3d) ------------------------------- #
    head_rel = PooledMLP(latent_dim, 3, args.hidden).to(device)
    train_head(head_rel, lambda i: head_rel(z_t[i]), tgt["rel"], train_idx, val_idx,
               epochs=args.epochs, bs=args.bs, lr=args.lr, device=device, rng=rng, name="ee-obj")
    rp = batched_forward(head_rel, lambda i: (z_t[i],), N, device).numpy()
    err = np.linalg.norm(rp[val_idx] - tgt["rel"].numpy()[val_idx], axis=1)
    m = vec_metrics(err)
    rows.append(dict(target="hand-object rel vec (ee-obj)", kind="vec3", n=len(val_idx),
                     onp_metric=f"median={m['median_cm']:.2f}cm  <5cm={m['frac_within_5cm']*100:.1f}%  <7cm={m['frac_within_7cm']*100:.1f}%",
                     **{f"onp_{k}": v for k, v in m.items()}))

    # ---- c. full state-oracle cost value c*(s) from [z_t, z_g] -------------- #
    head_c = PairPooledMLP(latent_dim, 1, args.hidden).to(device)
    train_head(head_c, lambda i: head_c(z_t[i], z_g[i]).reshape(-1), c_true,
               train_idx, val_idx, epochs=args.epochs, bs=args.bs, lr=args.lr,
               device=device, rng=rng, name="cost c*")
    cp = batched_forward(head_c, lambda i: (z_t[i], z_g[i]), N, device).reshape(-1).numpy()
    m = scalar_metrics(cp[val_idx], c_true.numpy()[val_idx])
    rows.append(dict(target="state-oracle cost c*(s) [z_t,z_g]", kind="scalar(m)", n=len(val_idx),
                     onp_metric=f"R²={m['r2']:.3f} MAE={m['mae']*100:.2f}cm median={m['median_ae']*100:.2f}cm (std={m['true_std']*100:.1f}cm)",
                     **{f"onp_{k}": v for k, v in m.items()}))

    # ---- d. contact indicator: from z_t alone AND from (z_t, z_t1) --------- #
    from sklearn.metrics import roc_auc_score
    c_lbl = tgt["contact"].numpy()
    base_rate = float(c_lbl[val_idx].mean())

    head_ct1 = PooledMLP(latent_dim, 1, args.hidden).to(device)
    train_head(head_ct1, lambda i: head_ct1(z_t[i]).reshape(-1), tgt["contact"],
               train_idx, val_idx, epochs=args.epochs, bs=args.bs, lr=args.lr,
               device=device, rng=rng, bce=True, name="contact|z_t")
    ct1 = batched_forward(head_ct1, lambda i: (z_t[i],), N, device).reshape(-1).numpy()
    auc1 = float(roc_auc_score(c_lbl[val_idx], ct1[val_idx]))
    rows.append(dict(target="contact ‖Δobj‖>5mm  | z_t alone", kind="AUROC", n=len(val_idx),
                     onp_metric=f"AUROC={auc1:.3f} (base rate={base_rate:.2f})",
                     onp_auroc=auc1, onp_base_rate=base_rate))

    z_t1 = D["z_t1"].to(device)
    head_ctp = PairPooledMLP(latent_dim, 1, args.hidden).to(device)
    train_head(head_ctp, lambda i: head_ctp(z_t[i], z_t1[i]).reshape(-1), tgt["contact"],
               train_idx, val_idx, epochs=args.epochs, bs=args.bs, lr=args.lr,
               device=device, rng=rng, bce=True, name="contact|z_t,z_t1")
    ctp = batched_forward(head_ctp, lambda i: (z_t[i], z_t1[i]), N, device).reshape(-1).numpy()
    aucp = float(roc_auc_score(c_lbl[val_idx], ctp[val_idx]))
    rows.append(dict(target="contact ‖Δobj‖>5mm  | (z_t,z_{t+1})", kind="AUROC", n=len(val_idx),
                     onp_metric=f"AUROC={aucp:.3f} (base rate={base_rate:.2f})",
                     onp_auroc=aucp, onp_base_rate=base_rate))

    # ------------------------------------------------------------------ #
    # Off-policy pass (GPU): re-eval the single-frame probes on random-action
    # rollout latents — the distribution the CEM planner actually scores.
    # ------------------------------------------------------------------ #
    if args.offpolicy:
        off_rows = run_offpolicy(args, cfg, device, head_g, head_rel, latent_dim)

    # ---------------------------- write ------------------------------- #
    _write_outputs(args, rows, off_rows, N, len(val_idx))

    # save probe checkpoints (so a GPU job can re-load for off-policy)
    ck = Path(args.ckpt_dir); ck.mkdir(parents=True, exist_ok=True)
    torch.save({"gripper": head_g.state_dict(), "rel": head_rel.state_dict(),
                "cost": head_c.state_dict(), "contact_zt": head_ct1.state_dict(),
                "contact_pair": head_ctp.state_dict(), "latent_dim": latent_dim,
                "hidden": args.hidden},
               ck / f"encoder_info_upperbound_{args.model}.pt")
    print(f"\nwrote {args.out_csv}, {args.out_md}, and probe checkpoints", flush=True)
    return 0


def run_offpolicy(args, cfg, device, head_g, head_rel, latent_dim):
    """Collect random-action rollout frames, encode, eval gripper + ee-obj + object."""
    from models.adapters import build_adapter
    from scripts._offpolicy_frames import collect_offpolicy_frames
    print("\n=== off-policy pass: collecting random-action rollout frames ===", flush=True)
    adapter = build_adapter(args.model, device=str(device)).eval()
    buf = collect_offpolicy_frames(adapter, device, tasks=args.op_tasks,
                                   episodes=args.op_episodes, max_steps=args.op_max_steps,
                                   collect_every=args.op_collect_every, seed=30000,
                                   return_frames=True)
    z = buf["z"].to(device); obj = buf["obj"]; ee = buf["ee"]
    grip = buf["prop"][:, GRIPPER_IDX].float()               # obs[:4][3] = gripper aperture
    rel_true = (ee - obj)
    out = {}
    with torch.no_grad():
        gp = head_g(z).reshape(-1).cpu()
        rp = head_rel(z).cpu()
    out["gripper_aperture(state[3])"] = scalar_metrics(gp.numpy(), grip.numpy())
    err = np.linalg.norm(rp.numpy() - rel_true.numpy(), axis=1)
    out["hand-object rel vec (ee-obj)"] = vec_metrics(err)
    print(f"off-policy n={z.shape[0]}  gripper R²={out['gripper_aperture(state[3])']['r2']:.3f}  "
          f"ee-obj median={out['hand-object rel vec (ee-obj)']['median_cm']:.2f}cm", flush=True)
    return out


def _write_outputs(args, rows, off_rows, n_all, n_val):
    import pandas as pd
    # attach off-policy columns where available
    for r in rows:
        key = r["target"].split("  |")[0].strip()
        if r["target"] in off_rows or key in off_rows:
            o = off_rows.get(r["target"], off_rows.get(key))
            if r["kind"] == "scalar":
                r["offp_metric"] = f"R²={o['r2']:.3f} MAE={o['mae']:.3f} (nMAE={o['nmae']:.2f})"
            else:
                r["offp_metric"] = (f"median={o['median_cm']:.2f}cm <5cm={o['frac_within_5cm']*100:.1f}%")
        else:
            r["offp_metric"] = "on-policy only"
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    lines = []
    lines.append("# Encoder information upper-bound — dino_wm_metaworld frozen latent\n")
    lines.append(f"Held-out trajectory split (val_frac={args.val_frac}, seed={args.seed}); "
                 f"n_val_transitions={n_val}. Probes = small mean+max-pooled MLPs on the "
                 f"FROZEN latent; encoder + predictor untouched. Cost target and goal follow "
                 f"the state-oracle winner (scripts/29): c*(s)=‖obj−g‖+0.5·‖hand−obj‖, "
                 f"g = object at the trajectory's terminal (goal) frame, z_g = that frame's latent.\n")
    lines.append("## What the frozen latent carries (on-policy = expert cache; "
                 "off-policy = random-action rollout latents the planner scores)\n")
    lines.append("| target | metric type | on-policy | off-policy |")
    lines.append("|---|---|---|---|")
    # reference rows from existing results (object / ee) for a complete table
    lines.append("| object xyz (state[4:7]) *(scripts/21,34)* | vec3 | median 2.12cm, 92.1% <5cm | "
                 "robust probe: median 2.01cm, 91.5% <5cm (naive 3.44cm, 78.3%) |")
    lines.append("| end-effector xyz (state[0:3]) *(scripts/19,34)* | vec3 | median ~2cm | "
                 "median 5.14cm, 45.8% <5cm |")
    for r in rows:
        lines.append(f"| {r['target']} | {r['kind']} | {r['onp_metric']} | {r['offp_metric']} |")
    lines.append("")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    # echo table to stdout
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    sys.exit(main())
