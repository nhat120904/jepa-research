"""CLI for the #1-bug guard: validate action normalization on a real transition.

Run this on the server AFTER smoke_test passes and BEFORE trusting any metric:

    python scripts/check_normalization.py --config configs/diagnostic_metaworld.yaml \
        --model jepa_wm_metaworld --ref-eval-loss 0.12

``--ref-eval-loss`` is the model's reported eval MSE (from the JEPA-WMs paper /
checkpoint logs). If the factual prediction MSE is > ~2× this, normalization is
wrong. Implementation lives in ``sanity_check.check_action_normalization``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sanity_check import check_action_normalization  # noqa: E402


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--ref-eval-loss", type=float, default=None)
    args = p.parse_args()
    res = check_action_normalization(args.model, args.config, n_transitions=args.n,
                                     ref_eval_loss=args.ref_eval_loss)
    sys.exit(0 if res["ok"] else 1)
