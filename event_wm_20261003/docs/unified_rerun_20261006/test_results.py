"""A partial or duplicate episode set cannot be reported as the complete protocol."""
import importlib.util
assert importlib.util.find_spec('result_checks') is not None, 'Missing full-protocol result validation'
from result_checks import validate_episode_grid
full=[{'task':t,'episode':i,'success':True} for t in range(1,6) for i in range(20)]
validate_episode_grid(full)
for bad in (full[:-1],full[:-1]+[full[0]]):
    try:
        validate_episode_grid(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete/duplicate evaluation accepted as100episodes')
print('COMPLETE_EPISODE_PROTOCOL_OK')
