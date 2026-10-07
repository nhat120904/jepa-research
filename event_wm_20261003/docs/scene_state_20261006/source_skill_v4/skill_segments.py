"""BC segment eligibility, with optional multi-object event supervision."""

def eligible_segment(knock, include_side_effects, event_end, release, n):
    return bool((include_side_effects or not knock) and event_end + release < n)
