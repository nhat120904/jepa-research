"""GPC-style registration objective in the pinned gym-pusht coordinate system.

Privileged poses are used for training labels and the explicitly named GEOM8
oracle only. Learned CTA inference continues to see images and agent proprio.
"""
import numpy as np

GOAL_POSE = np.array([256., 256., np.pi / 4])
VERTICES = np.array([[-60., 30.], [60., 30.], [60., 0.], [-60., 0.],
                     [-15., 30.], [-15., 120.], [15., 120.], [15., 30.]])


def world_vertices(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1] != 3 or not np.isfinite(pose).all():
        raise ValueError('Expected finite (...,3) block poses')
    c, s = np.cos(pose[..., 2]), np.sin(pose[..., 2])
    x = c[..., None] * VERTICES[:, 0] - s[..., None] * VERTICES[:, 1] + pose[..., 0, None]
    y = s[..., None] * VERTICES[:, 0] + c[..., None] * VERTICES[:, 1] + pose[..., 1, None]
    return np.stack((x, y), axis=-1)


def registration_score(pose, goal=GOAL_POSE):
    """Higher is better; mean corresponding-vertex distance / 512, negated.

    Same ordering as a positively scaled sum of vertex distances. Native local
    vertices/rotation are used rather than the other GPC environment's geometry.
    The existing rank margin 1e-3 means 0.512 pixels of mean distance here.
    """
    return -np.linalg.norm(world_vertices(pose) - world_vertices(goal), axis=-1).mean(-1) / 512.


def native_geometry_check():
    """Compute node only: verify analytic labels against native transforms."""
    from ti_wm.contract import require_compute
    from gym_pusht.envs.pusht import PushTEnv
    require_compute()
    env = PushTEnv(obs_type='state')
    try:
        env.reset(seed=30250)
        np.testing.assert_allclose(env.goal_pose, GOAL_POSE, atol=1e-12)
        actual = np.array([tuple(v) for shape in env.block.shapes for v in shape.get_vertices()])
        if sorted(map(tuple, actual)) != sorted(map(tuple, VERTICES)):
            raise ValueError('Pinned native shape changed')
        max_error = 0.
        for theta in (-2.1, 0., .7, 2.8):
            env.block.angle = theta
            env.block.position = (190., 320.)
            pose = np.array([*env.block.position, env.block.angle])
            reference = np.array([tuple(env.block.local_to_world(tuple(v))) for v in VERTICES])
            error = float(np.abs(world_vertices(pose) - reference).max())
            max_error = max(max_error, error)
        if max_error > 1e-8:
            raise ValueError(f'Native vertex transform mismatch: {max_error}')
        return {'max_vertex_error': max_error, 'goal_pose': GOAL_POSE.tolist(), 'vertices': VERTICES.tolist()}
    finally:
        env.close()
