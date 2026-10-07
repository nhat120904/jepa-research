def validate_episode_grid(episodes, tasks=(1,2,3,4,5), per_task=20):
    expected={(t,i) for t in tasks for i in range(per_task)}
    observed=[(e['task'],e['episode']) for e in episodes]
    if len(observed)!=len(expected) or set(observed)!=expected:
        raise ValueError('Partial or duplicate episode set; full evaluation has not completed')
