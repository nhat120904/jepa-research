# GFPR H0-A

This folder implements the falsification-first pilot for **Gated Frozen-Pool
Physical Reranking (GFPR)**.  It is independent of the failed CROD method and
does not modify the frozen LeWM checkpoint.

The pilot asks a narrow question: can a scorer trained with exact same-state
physical regret labels select a better action chunk from a native LeWM final
CEM population on episode-held-out snapshots?  The scorer is never allowed to
refit CEM.  Its candidate distribution is frozen before it is consulted.

See `PROTOCOL.md` for the locked split, arms, and decision rule and
`JOB_LEDGER.md` for execution provenance.

