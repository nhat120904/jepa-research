import sys
from pathlib import Path

# Make the diagnosis package root importable (metrics, models, data, stratification).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
