import gym

import environments


def get_benchmark(env_id):
    """Verify that the specified env is amongst the admissible ones"""
    for k, v in environments.BENCHMARKS.items():
        if env_id in v:
            benchmark = k
            continue
    assert benchmark is not None, "unsupported environment"
    return benchmark


def make_env(env_id, seed):
    """Create an environment"""
    benchmark = get_benchmark(env_id)
    env = gym.make(env_id)
    env.seed(seed)
    if benchmark == 'mujoco':
        pass  # weird, but struct kept general if adding other envs
    else:
        raise ValueError('unsupported benchmark')
    return env
