"""An old event's release supervision cannot cross into a new episode."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'source'))
import skill_segments

assert hasattr(skill_segments, 'segment_frames'), 'BC release frames are not clipped at episode resets'
assert list(skill_segments.segment_frames(40, 49, 10, 50)) == list(range(40, 51))
assert list(skill_segments.segment_frames(40, 49, 10, 100)) == list(range(40, 60))
assert list(skill_segments.segment_frames(49, 49, 10, 49)) == [49]
assert skill_segments.eligible_segment(True, True, 49, 0, 50)
assert not skill_segments.eligible_segment(True, False, 49, 0, 50)
print('BC_EPISODE_RELEASE_BOUNDARY_OK')
