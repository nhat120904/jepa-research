"""Selector bookkeeping of scripts/libero/l4_continuation.py (numpy only)."""
import importlib.util
import unittest
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "scripts" / "libero" / "l4_continuation.py"
_spec = importlib.util.spec_from_file_location("l4", _path)
l4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l4)


def cand(progress, cont):
    return {"progress_end": progress, "cont": cont}


class AnchorStats(unittest.TestCase):
    def test_heldout_streams_only(self):
        # candidate 3 looks best on selection streams 0-1 but fails on 2-3: held-out gain must be 0 - ev[0]
        c = [cand(0.1, [0, 0, 1, 1])] + [cand(0.1, [0, 0, 0, 0])] * 2 + [cand(0.1, [1, 1, 0, 0])] + \
            [cand(0.1, [0, 0, 0, 0])] * 4
        s = l4.anchor_stats({"candidates": c})
        self.assertEqual(s["cont_select"], -1.0)
        self.assertEqual(s["progress_select"], 0.0)  # all progress tied -> candidate 0

    def test_progress_selector_and_ties(self):
        c = [cand(0.5, [0, 0, 0, 0]), cand(0.9, [1, 1, 1, 1]), cand(0.51, [1, 1, 1, 1])] + \
            [cand(0.0, [0, 0, 0, 0])] * 5
        s = l4.anchor_stats({"candidates": c})
        self.assertEqual(s["progress_select"], 1.0)
        self.assertEqual(s["cont_select"], 1.0)
        n, differ = l4.endpoint_ties({"candidates": c})
        self.assertEqual(n, 1 + 10)  # (0,2) plus the 5 tied zeros
        self.assertEqual(differ, 1)


if __name__ == "__main__":
    unittest.main()
