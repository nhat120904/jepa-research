"""A resting side effect must not end an event while the acted object is in transit."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / 'source_boundary_v3'))
from event_boundary import acted_event_finished

def test_wait_for_the_acted_object_to_rest():
    changed = np.array([True, False, False, True, False])
    settled = np.array([False, True, True, True, True])
    assert not acted_event_finished(0, changed, settled, arrived=False)
    settled[0] = True
    assert acted_event_finished(0, changed, settled, arrived=False)
    changed[:] = False
    assert acted_event_finished(0, changed, settled, arrived=True)
    assert not acted_event_finished(0, changed, settled, arrived=False)

if __name__ == '__main__':
    test_wait_for_the_acted_object_to_rest()
    print('ACTED_EVENT_BOUNDARY_OK')
