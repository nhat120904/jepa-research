"""K-scaling analysis of a DRAW=64 PushT run (scripts/slurm_cta_v2_closed.sh, arms P0 and GEOM64). CPU only.

Question: with more samples from the SAME policy (no perturbation), does the gain a learned scorer realizes grow, and
does CTA's lead over the direct / endpoint / DINO-WM scorers grow with it?

For each bank size K in {8, 16, 32, 64} the bank is the first K candidates of the logged 64 (candidate k is seeded by
(root, decision, k) only, so K = 8 is the 55666 bank). Per scorer s and state set:
  gain_s(K)     = sum_d [y(argmax_{k<K} s) - y(0)]  per root, reported per decision (geometry label; coverage too)
  retained_s(K) = gain_s(K) / oracle gain(K)
  capture_s(K)  = share of success crossings (some k<K has coverage > .95, candidate 0 does not) where s picks a hit
  worse_s(K)    = share of decisions where s picks a candidate > .005 geometry worse than candidate 0
Root-cluster bootstrap CIs for gain_s(64) - gain_s(8) and for scorer differences at each K.

Decision rule, fixed before the run is read (docs: JOB_LEDGER 2026-09-29): a closed loop of learned arms at K = 64 is
run only if, on P0 states, (a) CTAV2's gain at K = 64 exceeds its gain at K = 8 (bootstrap CI of the difference above
0), and (b) CTAV2 - DIRV2L at K = 64 has a CI above 0. CTAV2S (answers averaged over 8 sampled codes) and the
uncertainty-penalized CTAV2S - b * CTAV2SD are exploratory here; any b chosen on these roots needs a fresh closed loop.
"""
import argparse
import json
from pathlib import Path

import numpy as np

THR, WORSE, B = .95, .005, 4000
KS = (8, 16, 32, 64)
PAIRS = ("CTAV2-DIRV2L", "CTAV2-DIRV2", "CTAV2-ENDV2", "CTAV2-DINOWM", "CTAV2S-CTAV2", "CTAV2S-DIRV2L",
         "CTAV2-CTA4", "FULLV2-CTAV2")


def load(run):
    shards = sorted(Path(run).glob("shard_*"))
    rep = json.loads((shards[0] / "closed_report.json").read_text())
    logs, eps = {}, {}
    for sh in shards:
        r = json.loads((sh / "closed_report.json").read_text())
        if r["status"] != "DONE" or r["arms"] != rep["arms"] or r["hashes"] != rep["hashes"]:
            raise ValueError(f"bad shard {sh}")
        for a in rep["arms"]:
            with np.load(sh / f"log_{a}.npz") as z:
                for k in z.files:
                    logs.setdefault(a, {}).setdefault(k, []).append(z[k])
        for line in (sh / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            eps.setdefault(e["arm"], {})[e["root"]] = e
    return rep, {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in logs.items()}, eps


def scorer_rows(L, names):
    s = {n: L[f"score_{n}"] for n in names if f"score_{n}" in L}
    if "CTAV2S" in s and "CTAV2SD" in s:
        for b in (.5, 1., 2.):
            s[f"CTAV2S-{b}sd"] = s["CTAV2S"] - b * s["CTAV2SD"]
    s.pop("CTAV2SD", None)
    return s


def per_root(L, scores, k):
    y, cov, roots = L["geom"][:, :k], L["cov"][:, :k], L["root"]
    uniq, inv = np.unique(roots, return_inverse=True)
    rows, R = np.arange(len(y)), len(uniq)
    hit = cov > THR
    cross = hit.any(1) & ~hit[:, 0]
    out = {"oracle": np.bincount(inv, y.max(1) - y[:, 0], R), "cross": np.bincount(inv, cross, R),
           "n": np.bincount(inv, np.ones(len(y)), R), "gain": {}, "cap": {}, "worse": {}}
    for n, s in scores.items():
        c = np.argmax(s[:, :k], 1)
        out["gain"][n] = np.bincount(inv, y[rows, c] - y[:, 0], R)
        out["cap"][n] = np.bincount(inv, hit[rows, c] & cross, R)
        out["worse"][n] = np.bincount(inv, y[rows, c] < y[:, 0] - WORSE, R)
    return out


def ci(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def analyse(L, names, seed=0):
    scores = scorer_rows(L, names)
    parts = {k: per_root(L, scores, k) for k in KS}
    R = len(parts[8]["n"])
    idx = np.random.default_rng(seed).integers(0, R, (B, R))
    n_dec = parts[8]["n"].sum()
    res = {"decisions": int(n_dec), "roots": int(R), "by_k": {}, "k64_minus_k8": {}, "pairs": {}}
    for k, p in parts.items():
        row = {"oracle_gain_per_decision": float(p["oracle"].sum() / n_dec), "crossings": int(p["cross"].sum()),
               "scorers": {}}
        for n in scores:
            row["scorers"][n] = {"gain_per_decision": float(p["gain"][n].sum() / n_dec),
                                 "retained": float(p["gain"][n].sum() / max(p["oracle"].sum(), 1e-12)),
                                 "capture": float(p["cap"][n].sum() / max(p["cross"].sum(), 1)),
                                 "worse_than_p0": float(p["worse"][n].sum() / n_dec)}
        res["by_k"][k] = row
        for pair in PAIRS:
            a, b = pair.split("-", 1)
            if a in scores and b in scores:
                d = (p["gain"][a] - p["gain"][b])[idx].sum(1) / parts[8]["n"][idx].sum(1)
                res["pairs"].setdefault(pair, {})[k] = [float((p["gain"][a] - p["gain"][b]).sum() / n_dec)] + ci(d)
    for n in list(scores) + ["oracle"]:
        g64 = parts[64]["oracle"] if n == "oracle" else parts[64]["gain"][n]
        g8 = parts[8]["oracle"] if n == "oracle" else parts[8]["gain"][n]
        d = (g64 - g8)[idx].sum(1) / parts[8]["n"][idx].sum(1)
        res["k64_minus_k8"][n] = [float((g64 - g8).sum() / n_dec)] + ci(d)
    return res


def main(a):
    rep, logs, eps = load(a.closed_run)
    names = rep["log_scorers"]
    out = {"run": str(a.closed_run), "draw": rep.get("draw"), "arms": rep["arms"], "sets": {},
           "success": {arm: int(sum(e["success"] for e in d.values())) for arm, d in eps.items()},
           "roots": len(eps["P0"])}
    if a.reference_run:
        # candidates 0-7 at the same (root, decision) must reproduce the 55666 K-bank labels while P0 states agree
        _, ref, ref_eps = load(a.reference_run)
        mine, theirs = logs["P0"], ref["P0"]
        key = lambda L: {(int(r), int(d)): i for i, (r, d) in enumerate(zip(L["root"], L["decision"]))}
        km, kt = key(mine), key(theirs)
        common = sorted(set(km) & set(kt))
        same = [np.allclose(mine["geom"][km[c], :8], theirs["geom"][kt[c]], atol=1e-4) for c in common]
        first = [c for c in common if c[1] == 0]
        same0 = [np.allclose(mine["geom"][km[c], :8], theirs["geom"][kt[c]], atol=1e-4) for c in first]
        out["reference"] = {"common_decisions": len(common), "k_bank_identical": float(np.mean(same)),
                            "decision0_identical": float(np.mean(same0)),
                            "p0_success_here_vs_ref": [out["success"]["P0"],
                                                       int(sum(e["success"] for e in ref_eps["P0"].values()))],
                            "ref_geom8_success": int(sum(e["success"] for e in ref_eps.get("GEOM8", {}).values()))}
        # < 1 at decision 0 would mean the seed mapping differs (unit-tested) or batch-size numerics moved the policy
        # draw; later decisions may differ once P0's own trajectory diverges. Reported, not fatal.
    for arm in rep["arms"]:
        out["sets"][f"{arm}_states"] = analyse(logs[arm], names)
    p = out["sets"]["P0_states"]
    gate_a = p["k64_minus_k8"]["CTAV2"][1] > 0
    gate_b = "CTAV2-DIRV2L" in p["pairs"] and p["pairs"]["CTAV2-DIRV2L"][64][1] > 0
    out["decision"] = {"a_ctav2_gain_grows": bool(gate_a), "b_ctav2_beats_dirv2l_at_64": bool(gate_b),
                       "run_closed_loop_at_64": bool(gate_a and gate_b)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "kscale.json").write_text(json.dumps(out, indent=1))
    lines = [f"success {out['success']} (roots {out['roots']})", json.dumps(out.get("reference")), ""]
    for key_, r in out["sets"].items():
        lines += [f"## {key_}: {r['decisions']} decisions, {r['roots']} roots", "",
                  "| scorer | " + " | ".join(f"gain K{k} (ret, cap, worse)" for k in KS) + " | K64-K8 [CI] |",
                  "|---|" + "---|" * (len(KS) + 1)]
        orc = " | ".join(f"{r['by_k'][k]['oracle_gain_per_decision']*1e3:.2f} (1, {r['by_k'][k]['crossings']} cr)"
                         for k in KS)
        d = r["k64_minus_k8"]["oracle"]
        lines.append(f"| oracle | {orc} | {d[0]*1e3:+.2f} [{d[1]*1e3:+.2f}, {d[2]*1e3:+.2f}] |")
        for n in r["by_k"][8]["scorers"]:
            cells = []
            for k in KS:
                v = r["by_k"][k]["scorers"][n]
                cells.append(f"{v['gain_per_decision']*1e3:.2f} ({v['retained']:.2f}, {v['capture']:.2f}, "
                             f"{v['worse_than_p0']:.3f})")
            d = r["k64_minus_k8"][n]
            lines.append(f"| {n} | " + " | ".join(cells) + f" | {d[0]*1e3:+.2f} [{d[1]*1e3:+.2f}, {d[2]*1e3:+.2f}] |")
        lines += ["", "| pair (gain x1e3 per decision) | " + " | ".join(f"K{k}" for k in KS) + " |",
                  "|---|" + "---|" * len(KS)]
        for pair, v in r["pairs"].items():
            lines.append(f"| {pair} | " + " | ".join(f"{v[k][0]*1e3:+.2f} [{v[k][1]*1e3:+.2f}, {v[k][2]*1e3:+.2f}]"
                                                     for k in KS) + " |")
        lines.append("")
    lines.append(f"decision: {out['decision']}")
    (a.out / "kscale.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--closed-run", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
