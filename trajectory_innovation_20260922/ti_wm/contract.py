"""Small deterministic contracts; no simulator or model imports."""

import hashlib
import math
import os

LEROBOT_REVISION = "3c0a209f9fac4d2a57617e686a7f2a2309144ba2"
POLICY_REPO = "lerobot/diffusion_pusht"
BANK_SIZES = (8, 16, 32)


def require_compute():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run through sbatch on a compute node")


def candidate_seed(root, decision, candidate):
    if min(root, decision, candidate) < 0:
        raise ValueError("Seed coordinates must be nonnegative")
    # Bank size is intentionally absent: increasing K preserves the existing bank.
    message = f"ti-v1/{root}/{decision}/{candidate}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little") % (2**63 - 1)


def _hash_int(message):
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little") % (2**63 - 1)


def continuation_seed(root, candidate, rep, decision):
    """Seed for the policy draws that continue a branch after its first chunk (H_rep)."""
    if min(root, candidate, rep, decision) < 0:
        raise ValueError("Seed coordinates must be nonnegative")
    return _hash_int(f"ti-v1-cont/{root}/{candidate}/{rep}/{decision}")


def sibling_seed(root, decision, candidate, chunk):
    """Seed for the policy draws that continue sibling `candidate` of a decision (gate S1)."""
    if min(root, decision, candidate, chunk) < 0:
        raise ValueError("Seed coordinates must be nonnegative")
    return _hash_int(f"ti-v1-sib/{root}/{decision}/{candidate}/{chunk}")


def anchor_fraction(root):
    """Outcome-independent position of the H_rep anchor along a root's P0 trajectory."""
    return (_hash_int(f"ti-v1-anchor/{root}") % 1_000_000) / 1_000_000


def medoid_index(chunks):
    """Candidate whose executed chunk has the smallest summed L2 distance to the others."""
    flat = [[float(v) for step in chunk for v in step] for chunk in chunks]
    if not flat:
        raise ValueError("Empty bank")
    totals = [sum(math.dist(a, b) for b in flat) for a in flat]
    return min(range(len(totals)), key=lambda i: totals[i])


def native_action_slice(config):
    start = config["n_obs_steps"] - 1
    end = start + config["n_action_steps"]
    if start < 0 or config["n_action_steps"] <= 0 or end > config["horizon"]:
        raise ValueError("Invalid native action alignment")
    return start, end


def select_candidate(scores):
    """Highest score; default (index 0) wins if tied for best, otherwise first best."""
    if not scores or not all(math.isfinite(x) for x in scores):
        raise ValueError("Scores must be nonempty and finite")
    return max(range(len(scores)), key=lambda i: scores[i])


def compatible_config(raw, allowed_fields):
    removed = {k: raw[k] for k in ("device", "use_amp") if k in raw and k not in allowed_fields}
    config = {k: v for k, v in raw.items() if k not in removed and k != "type"}
    unknown = set(config) - set(allowed_fields)
    if unknown:
        raise ValueError(f"Unknown policy fields: {sorted(unknown)}")
    if raw.get("type") != "diffusion":
        raise ValueError("Wrong policy type")
    return config, removed
