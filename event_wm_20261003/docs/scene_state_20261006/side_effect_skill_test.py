"""Intentional coupled motion is eligible for event-conditioned BC supervision."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'source_skill_v4'))
from skill_segments import eligible_segment

assert eligible_segment(knock=True, include_side_effects=True, event_end=50, release=10, n=100)
assert not eligible_segment(knock=True, include_side_effects=False, event_end=50, release=10, n=100)
assert eligible_segment(knock=False, include_side_effects=True, event_end=50, release=10, n=100)
assert not eligible_segment(knock=False, include_side_effects=True, event_end=95, release=10, n=100)
print('COUPLED_SKILL_SEGMENT_SEMANTICS_OK')
