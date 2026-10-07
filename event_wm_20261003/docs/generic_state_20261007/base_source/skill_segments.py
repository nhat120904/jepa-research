"""BC segment eligibility, with optional multi-object event supervision."""

def eligible_segment(knock, include_side_effects, event_end, release, n):
    return bool((include_side_effects or not knock) and event_end + release < n)


def segment_frames(start, event_end, release, episode_last):
    if not 0 <= start <= event_end <= episode_last:
        raise ValueError('Event segment must lie within its episode')
    return range(int(start), min(int(event_end + release), int(episode_last)) + 1)
