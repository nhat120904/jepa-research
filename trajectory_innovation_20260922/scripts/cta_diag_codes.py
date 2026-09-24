"""Code-usage diagnostic for a trained CTA checkpoint (docs/CTA_DEBUG_LOG.md). CPU job, dev split only.

Per FSQ dimension: histogram of quantization levels, fraction at the two extreme levels, and the distribution of the
pre-bound activation |z + shift| (tanh saturation shows as large values and extreme-level mass).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ti_wm.contract import require_compute


def main(run, features, n):
    require_compute()
    import importlib.util
    spec = importlib.util.spec_from_file_location("cta_train", Path(__file__).with_name("cta_train.py"))
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    blob = torch.load(run / "cta.pt", map_location="cpu")
    models = ct.build(blob["config"], torch.device("cpu"))
    models["enc"].load_state_dict(blob["state"]["enc"])
    enc = models["enc"].eval()
    dev = ct.Split(features / "dev", n)
    zs = []
    hook = enc.to_code.register_forward_hook(lambda m, i, o: zs.append(o.detach()))
    with torch.inference_mode():
        for s in range(0, dev.n, 8):
            i = torch.arange(s, min(s + 8, dev.n))
            enc(dev.context(i, "cpu"), dev.future(i, "cpu"))
    hook.remove()
    z = torch.cat(zs).reshape(-1, zs[0].shape[-1])
    fsq = enc.fsq
    shift = torch.atanh(fsq.offset / fsq.half_l)
    pre = z + shift
    digits = torch.round((torch.tanh(pre) * fsq.half_l - fsq.offset) + fsq.half_w).long()
    out = {}
    for d in range(z.shape[-1]):
        lv = int(fsq.levels[d])
        hist = np.bincount(digits[:, d].numpy(), minlength=lv) / len(digits)
        out[f"dim{d}"] = {"levels": lv, "hist": hist.round(4).tolist(), "extreme_mass": float(hist[0] + hist[-1]),
                          "abs_pre_quantiles": np.quantile(pre[:, d].abs().numpy(), [0.5, 0.9, 0.99]).round(3).tolist()}
    print(json.dumps(out, indent=1))
    (run / "code_usage_diag.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--n", type=int, default=400)
    a = p.parse_args()
    main(a.run, a.features, a.n)
