
## 2026-10-06 cost-to-go example for the existing demo

- Job 57640, ew_costdemo: main CPU, 2 CPU, 8 GB, 2-minute limit. Recompute h(S,G) on the four saved task-2 states in demo report job57580, using the same model SHA256 7eb88a9dd077195f809c98afaefa505d90364faf584d26a03f9e707df3f4bcd8 and frozen u_wm.py from demo57497. No training, search rerun, rendering, physics, or model changes.
- Source: docs/demos/cost_search_example_20261006/{read_cost.py,read_cost.sbatch}; remote distinct dir /mnt/data/nhatnc129/jepa/event_wm/cost_search_example_20261006. Values are CPU recomputations, not logged original priorities. Status at submission: SUBMITTED.
