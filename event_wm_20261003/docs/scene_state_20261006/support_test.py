"""Check a conservative learned-event support wrapper without model loading."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / 'source_support_v2'))
import u_wm

def test_unsupported_transitions_do_not_create_imagined_progress():
    assert hasattr(u_wm, 'apply_event_support'), 'learned event support has not been implemented'
    S = np.zeros((3, 5, 6), np.float32)
    predicted = np.ones_like(S)
    gated = u_wm.apply_event_support(S, predicted, np.array([.1, .6, .9]), np.array([.5, .5, .5]))
    np.testing.assert_array_equal(gated[0], S[0])
    np.testing.assert_array_equal(gated[1:], predicted[1:])
    np.testing.assert_array_equal(S, 0.)

if __name__ == '__main__':
    test_unsupported_transitions_do_not_create_imagined_progress()
    print('LEARNED_SUPPORT_WRAPPER_OK')
