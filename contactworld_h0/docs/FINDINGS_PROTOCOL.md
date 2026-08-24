# ContactWorld — protocol facts established by reading the released code

Date: 2026-08-20. Source: `PokuangZhou/ContactWorld` @ main, `purdue-mars/manifeel-isaacgymenvs` @ main.

## 1. What "success rate" actually measures

It is **not** semantic task completion. It is **demo-goal pose reaching**.

- Goals are sampled as `(episode, start_step, start_step + goal_offset_steps)` over the
  demonstration dataset (`eval_planner.py:188-202`). The env is reset to the demo's state at
  `start_step`; the target is the demo's state `goal_offset_steps` later.
- Success requires **four** simultaneous conditions (`eval_planner.py:299-315`):

      plug_pos_err      < 0.01 m      plug_quat_err < 15 deg
      ee_pos_err        < 0.01 m      ee_quat_err   < 15 deg

  and `success = plug_success AND ee_success`.
- `plug_success` and `ee_success` are logged separately (`:1315-1316`), which makes a clean
  decomposition available: object-state failure vs. arm-control failure.

Consequence for us: the headline **36.1%** is "fraction of sampled demo segments where the
planner brought both the object and the end-effector within 1 cm / 15 deg of the demo state".
Any claim we make must be phrased against that, not against "task success".

## 2. Step budget is NOT a confound (checked)

`max_steps = 1.25 x goal_offset_steps` at every horizon (12/15, 24/30, 36/45, 48/60), with
`frameskip=1` and one `env.step` per replan. So the long-horizon collapse is not caused by
starving the planner of environment steps. This one is clean.

## 3. Protocol detail: the gripper channel is replayed from the demonstration

`eval_planner.py:1269-1279`:

    if action_dim > 6:
        gt_actions_step = dataset.get_col_data("action")[gt_action_rows]
        action_np[active_mask, 6] = gt_actions_step[active_mask, 6]

The 7th action dimension (gripper) is **overwritten with the ground-truth demo action** at every
step. The planner never controls grasp timing; it is replayed from the demo that generated the
goal. All published numbers are under this oracle-gripper schedule.

This is not necessarily wrong (it isolates reaching/contact control from grasp scheduling), but
it is unstated in the paper, it is privileged information entering the rollout, and any method we
propose inherits it. It must be disclosed in ours, and it removes "when to close the gripper"
from the space of things a method could improve.

## 4. exploration_search ("Blind Box") — what is actually hidden

Dataset: 108 episodes, 7844 steps, mean 72.6 steps/ep (min 2, max 103).

- `socket_pos == socket_pos_gt` **exactly**, and both are the constant `(0.5, 0, 0.01)` in every
  episode (std = 0 across and within episodes). The socket is *not* the hidden variable, and the
  `_gt` suffix carries no extra information in this task.
- The randomized factors across episodes are the **plug** initial pose:
  `plug_pos` init std = (0.030, 0.00002, 0.0025) m and `plug_quat` init std = (0.148, ..., 0.010).
- The plug moves within episodes and ends up *farther* from the socket than it started
  (0.049 m -> 0.120 m mean), i.e. this is not a plug-into-socket insertion task.

So the task-relevant hidden state to probe is the **plug pose**, which is exactly what the
success metric thresholds.

## 5. Object state is genuinely under-determined by proprioception

Least-squares readout of `plug_pos` from end-effector proprioception (`ee_pos`, `ee_quat`):

    residual mean 2.11 cm, median 1.66 cm; only 24.5% of frames within the 1 cm success threshold

The plug is therefore not rigidly slaved to the gripper — there is real object-state uncertainty
above the benchmark's own success tolerance. This is the precondition the BAT-WM premise needs,
and it holds.

Whether *tactile* is what resolves that uncertainty is a separate question, measured in
`01_observability_probe.py`.
