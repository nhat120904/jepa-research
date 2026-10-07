"""Generic rest-to-rest completion of the commanded event's acted object."""

def acted_event_finished(actor, changed, settled, arrived):
    # Other objects can settle during the acted object's ongoing manipulation.
    return bool(settled[actor] and (changed[actor] or arrived))
