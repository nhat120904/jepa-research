"""Append Part 3 (cross-dataset + scaling curve) to regime_visualization.ipynb,
executing each new cell headlessly and embedding its outputs — WITHOUT touching
the already-executed Metaworld/DROID frame galleries in Parts 1–2 (the cluster box
has no Metaworld parquet and no Jupyter, so a full re-execute would destroy them).

Idempotent: if a Part 3 already exists it is replaced. Run from diagnosis/:
    .venv/bin/python notebooks/_append_part3.py
"""
import sys, io, json, base64, contextlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _part3_cells import PART3  # noqa: E402

NB_PATH = HERE / "regime_visualization.ipynb"
PART3_MARKER = "# Part 3 · Other datasets"

# ---- output-capturing exec -------------------------------------------------
_ns = {}
_OUT = []
_stdout = io.StringIO()


def _flush_stdout():
    s = _stdout.getvalue()
    if s:
        _OUT.append({"output_type": "stream", "name": "stdout", "text": s.splitlines(keepends=True)})
        _stdout.truncate(0); _stdout.seek(0)


def _display(obj):
    _flush_stdout()
    data = {}
    html = getattr(obj, "_repr_html_", None)
    if callable(html):
        try:
            data["text/html"] = html()
        except Exception:
            pass
    data["text/plain"] = repr(obj)
    _OUT.append({"output_type": "display_data", "data": data, "metadata": {}})


def _show(*a, **k):
    _flush_stdout()
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        _OUT.append({"output_type": "display_data",
                     "data": {"image/png": b64}, "metadata": {}})
        plt.close(fig)


_ns["display"] = _display
plt.show = _show


def run_code(src):
    global _OUT
    _OUT = []
    _stdout.truncate(0); _stdout.seek(0)
    with contextlib.redirect_stdout(_stdout):
        try:
            exec(compile(src, "<part3>", "exec"), _ns)
        except Exception as e:
            import traceback
            _flush_stdout()
            _OUT.append({"output_type": "stream", "name": "stderr",
                         "text": traceback.format_exc().splitlines(keepends=True)})
            print(f"  [cell error] {type(e).__name__}: {e}", file=sys.__stderr__)
    _flush_stdout()
    return list(_OUT)


def build_cells():
    out_cells = []
    n_fig = n_err = 0
    for kind, src in PART3:
        if kind == "md":
            out_cells.append({"cell_type": "markdown", "metadata": {},
                              "source": src.splitlines(keepends=True)})
        else:
            outs = run_code(src)
            n_fig += sum(1 for o in outs if o.get("data", {}).get("image/png"))
            n_err += sum(1 for o in outs if o.get("name") == "stderr")
            out_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                              "outputs": outs, "source": src.splitlines(keepends=True)})
    return out_cells, n_fig, n_err


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    # drop any prior Part 3 (idempotent)
    cut = next((i for i, c in enumerate(cells)
                if c["cell_type"] == "markdown" and any(PART3_MARKER in ln for ln in c["source"])), None)
    if cut is not None:
        print(f"replacing existing Part 3 (from cell {cut})")
        cells = cells[:cut]
    new_cells, n_fig, n_err = build_cells()
    nb["cells"] = cells + new_cells
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH.name}: {len(cells)} kept + {len(new_cells)} Part-3 cells "
          f"({n_fig} figures embedded, {n_err} cell errors)")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
