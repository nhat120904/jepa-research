"""Decision-Equivalent Belief Compression + amortized decision-regret VOI.

An oracle-level feasibility gate (Gate B) for a POMDP planning method,
implemented on tiny, fully-controllable POMDPs in pure Python + numpy.

No neural networks, no robot, no physics engine. Everything here computes
EXACT oracle beliefs and TRUE (exact-expectation) regret so the algorithm
can be validated cheaply and decisively.
"""

__all__ = [
    "core",
    "tasks",
    "compression",
    "planners",
    "evaluate",
    "experiments",
    "scaling",
    "p0",
]
