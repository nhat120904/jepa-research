"""Scratch: pull the Metaworld planning-success protocol/numbers from the
upstream JEPA-WMs paper PDF (world_model/jepa-success.pdf)."""
import re
import sys

from pypdf import PdfReader

r = PdfReader(r"E:\code-project\jepa-research\world_model\jepa-success.pdf")
query = sys.argv[1] if len(sys.argv) > 1 else "etaworld"
for i, p in enumerate(r.pages):
    t = p.extract_text() or ""
    if re.search(query, t):
        print(f"\n===== page {i+1} =====")
        print(t[:4000])
