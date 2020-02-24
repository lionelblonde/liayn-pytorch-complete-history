# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MuJoCo environments.

MUJOCO_ROBOTS = [
    'InvertedPendulum',
    'InvertedDoublePendulum',
    'Reacher',
    'Hopper',
    'HalfCheetah',
    'Walker2d',
    'Ant',
    'Humanoid',
]

MUJOCO_ENVS = ["{}-v2".format(name) for name in MUJOCO_ROBOTS]
MUJOCO_ENVS.extend(["{}-v3".format(name) for name in MUJOCO_ROBOTS])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Aggregate the environments

BENCHMARKS = {
    'mujoco': MUJOCO_ENVS,
}
