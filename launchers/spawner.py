"""Example launch
    python -m algorithms.spawners.spawn_jobs \
        --task=ddpg \
        --benchmark=mujoco \
        --cluster=cscs \
        --num_workers=4 \
        --partition=shared-gpu \
        --time=12:00:00 \
        --num_seeds=5 \
        --no-call \
        --no-rand
"""

import argparse
import os.path as osp
import numpy as np
from subprocess import call
from copy import copy
import yaml

from algorithms.helpers.misc_util import zipsame, boolean_flag
from algorithms.helpers.experiment_initializer import rand_id


parser = argparse.ArgumentParser(description="SAM Job Orchestrator + HP Search")
parser.add_argument('--task', type=str, choices=['ddpg'], default='ddpg')
parser.add_argument('--benchmark', type=str, choices=['ale', 'mujoco'], default='mujoco')
parser.add_argument('--cluster', type=str, choices=['baobab', 'cscs', 'box'], default='box',
                    help="cluster on which the experiments will be launched")
parser.add_argument('--num_rand_trials', type=int, default=50,
                    help="number of different models to run for the HP search")
parser.add_argument('--num_workers', type=int, default=1,
                    help="number of parallel workers to use for each job")
parser.add_argument('--partition', type=str, default=None, help="partition to launch jobs on")
parser.add_argument('--time', type=str, default=None, help="duration of the jobs")
parser.add_argument('--num_seeds', type=int, default=5,
                    help="amount of seeds across which jobs are replicated ('range(num_seeds)'')")
boolean_flag(parser, 'call', default=False, help="whether to launch the jobs once created")
boolean_flag(parser, 'rand', default=False, help="whether to perform hyperparameter search")
args = parser.parse_args()

# Load the content of the config file
envs = yaml.load(open("admissible_envs.yml"))['environments']

MUJOCO_ENVS_SET = list(envs['mujoco'].keys())
ALE_ENVS_SET = list(envs['ale'].keys())
MUJOCO_EXPERT_DEMOS = list(envs['mujoco'].values())
ALE_EXPERT_DEMOS = list(envs['ale'].values())


def zipsame_prune(l1, l2):
    out = []
    for a, b in zipsame(l1, l2):
        if b is None:
            continue
        out.append((a, b))
    return out


def fmt_path(args, meta, dir_):
    """Transform a relative path into an absolute path"""
    return osp.join("data/{}".format(meta), dir_)


def dup_hps_for_env(hpmap, env):
    """Return a separate copy of the HP map after adding extra key-value pair
    for the key 'env_id'
    """
    hpmap_ = copy(hpmap)
    hpmap_.update({'env_id': env})
    return hpmap_


def dup_hps_for_seed(hpmap, seed):
    """Return a separate copy of the HP map after adding extra key-value pairs
    for the key 'seed'
    """
    hpmap_ = copy(hpmap)
    hpmap_.update({'seed': seed})
    return hpmap_


def rand_tuple_from_list(list_):
    """Return a random tuple from a list of tuples
    (Function created because `np.random.choice` does not work on lists of tuples)
    """
    assert all(isinstance(v, tuple) for v in list_), "not a list of tuples"
    return list_[np.random.randint(low=0, high=len(list_))]


def get_rand_hps(args, meta):
    """Return a list of maps of hyperparameters selected by random search
    Example of hyperparameter dictionary:
        {'hid_widths': rand_tuple_from_list([(64, 64)]),  # list of tuples
         'hid_nonlin': np.random.choice(['relu', 'leaky_relu']),
         'hid_w_init': np.random.choice(['he_normal', 'he_uniform']),
         'polyak': np.random.choice([0.001, 0.01]),
         'with_layernorm': 1,
         'ent_reg_scale': 0.}
    """
    if args.benchmark == 'bench':
        if args.task == 'task':
            hpmap = {}  # TODO
            return hpmap


def get_spectrum_hps(args, meta, num_seeds):
    """Return a list of maps of hyperparameters selected deterministically
    and spanning the specified range of seeds
    Example of hyperparameter dictionary:
        {'hid_widths': (64, 64),
         'hid_nonlin': 'relu',
         'hid_w_init': 'he_normal',
         'polyak': 0.001,
         'with_layernorm': 1,
         'ent_reg_scale': 0.}
    """
    if args.benchmark == 'mujoco':
        if args.task == 'ddpg':
            hpmap = {'a': 1}  # TODO
            hpmaps = [dup_hps_for_env(hpmap, env) for env in MUJOCO_ENVS_SET]

    # Duplicate every hyperparameter map of the list to span the range of seeds
    output = [dup_hps_for_seed(hpmap_, seed)
              for seed in range(num_seeds)
              for hpmap_ in hpmaps]
    return output


def unroll_options(hpmap):
    """Transform the dictionary of hyperparameters into a string of bash options"""
    indent = 4 * ' '  # indents are defined as 4 spaces
    base_str = ""
    no_value_keys = ['from_raw_pixels',
                     'rmsify_obs',
                     'sample_or_mode',
                     'render',
                     'with_layernorm',
                     'rmsify_rets',
                     'enable_popart',
                     'prioritized_replay',
                     'ranked',
                     'unreal',
                     'n_step_returns']
    for k, v in hpmap.items():
        if k in no_value_keys and v == 0:
            base_str += "{}--no-{} \\\n".format(indent, k)
            continue
        elif k in no_value_keys:
            base_str += "{}--{} \\\n".format(indent, k)
            continue
        if isinstance(v, tuple):
            base_str += "{}--{} ".format(indent, k) + " ".join(str(v_) for v_ in v) + ' \\\n'
            continue
        base_str += "{}--{}={} \\\n".format(indent, k, v)
    return base_str


def format_job_str(args, job_map, run_str):
    """Build the batch script that launches a job"""

    if args.cluster == 'baobab':
        # Set sbatch config
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('#SBATCH --job-name={}\n'
                            '#SBATCH --partition={}\n'
                            '#SBATCH --ntasks={}\n'
                            '#SBATCH --cpus-per-task=1\n'
                            '#SBATCH --time={}\n'
                            '#SBATCH --mem=32000\n')
        contraint = "COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
        bash_script_str += ('#SBATCH --gres=gpu:1\n'
                            '#SBATCH --constraint="{}"\n'.format(contraint))
        bash_script_str += ('\n')
        # Load modules
        bash_script_str += ('module load GCC/6.3.0-2.27\n')
        bash_script_str += ('module load CUDA\n')
        bash_script_str += ('\n')
        # Launch command
        bash_script_str += ('srun {}')

        return bash_script_str.format(job_map['job-name'],
                                      job_map['partition'],
                                      job_map['ntasks'],
                                      job_map['time'],
                                      run_str)[:-2]

    elif args.cluster == 'cscs':
        # Set sbatch config
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('#SBATCH --job-name={}\n'
                            '#SBATCH --partition={}\n'
                            '#SBATCH --ntasks={}\n'
                            '#SBATCH --cpus-per-task=1\n'
                            '#SBATCH --time={}\n'
                            '#SBATCH --constraint=gpu\n\n')
        # Load modules
        bash_script_str += ('module load daint-gpu\n')
        bash_script_str += ('\n')
        # Launch command
        bash_script_str += ('srun {}')

        return bash_script_str.format(job_map['job-name'],
                                      job_map['partition'],
                                      job_map['ntasks'],
                                      job_map['time'],
                                      run_str)[:-2]

    elif args.cluster == 'box':
        # Set header
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('# job name: {}\n\n')
        # Launch command
        bash_script_str += ('mpirun -np {} {}')
        return bash_script_str.format(job_map['job-name'],
                                      job_map['ntasks'],
                                      run_str)[:-2]

    else:
        raise NotImplementedError("cluster selected is not covered")


def format_exp_str(args, hpmap):
    """Build the experiment name"""
    hpmap_str = unroll_options(hpmap)
    # Parse task name
    if args.task == 'ddpg':
        script = "algorithms.ddpg.run"

    return "python -m {} \\\n{}".format(script, hpmap_str)


def get_job_map(args, meta, i, env, seed, type_exp):
    return {'ntasks': args.num_workers,
            'partition': args.partition,
            'time': args.time,
            'job-name': "{}_{}{}_{}_{}_s{}".format(meta,
                                                   type_exp,
                                                   i,
                                                   args.task,
                                                   env.split('-')[0],
                                                   seed)}


def run(args):
    """Spawn jobs"""

    assert args.call or args.cluster == 'box', "on box, need to call manually!"

    # Create meta-experiment identifier
    meta = rand_id()
    # Define experiment type
    if args.rand:
        type_exp = 'hpsearch'
    else:
        type_exp = 'sweep'
    # Get hyperparameter configurations
    if args.rand:
        # Get a number of random hyperparameter configurations
        hpmaps = [get_rand_hps(args, meta, args.num_seeds) for _ in range(args.num_rand_trials)]
        # Flatten into a 1-dim list
        hpmaps = [x for hpmap in hpmaps for x in hpmap]
    else:
        # Get the deterministic spectrum of specified hyperparameters
        hpmaps = get_spectrum_hps(args, meta, args.num_seeds)
    # Create associated task strings
    exp_strs = [format_exp_str(args, hpmap) for hpmap in hpmaps]
    if not len(exp_strs) == len(set(exp_strs)):
        # Terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again :)")
    # Create the job maps
    job_maps = [get_job_map(args,
                            meta,
                            i,
                            hpmap['env_id'],
                            hpmap['seed'],
                            type_exp)
                for i, hpmap in enumerate(hpmaps)]
    # Finally get all the required job strings
    job_strs = [format_job_str(args, jm, es) for jm, es in zipsame(job_maps, exp_strs)]

    # Spawn the jobs

    # if args.cluster == 'box':
    #     yaml_content = {'session_name': 'run-all', 'windows': []}
    #     for i, js in enumerate(job_strs):
    #         yaml_content['windows'].append({'window_name': i,
    #                                         'panes': [js]})
    #     # Dump the assembled tmux config into a yaml file
    #     job_name = "{}_{}.yml".format(meta, type_exp)
    #     with open(job_name, "w") as f:
    #         yaml.dump(yaml_content, f, default_flow_style=False)

    # else:

    for i, (jm, js) in enumerate(zipsame(job_maps, job_strs)):
        print('-' * 10 + "> job #{} launcher content:".format(i))
        print(js + "\n")
        job_name = "{}.sh".format(jm['job-name'])
        with open(job_name, 'w') as f:
            f.write(js)
        if args.call:
            # Spawn the job!
            call(["sbatch", "./{}".format(job_name)])
    # Summarize the number of jobs spawned
    print("total num job (successfully) spawned: {}".format(len(job_strs)))

    if args.cluster == 'box':
        yaml_content = {'session_name': "{}_{}".format(meta, type_exp), 'windows': []}
        for i, jm in enumerate(job_maps):
            command = "./{}.sh".format(jm['job-name'])
            single_pane = {'shell_command': ["source activate pytorch-gym", command]}
            yaml_content['windows'].append({'window_name': str(i), 'panes': [single_pane]})
        # Dump the assembled tmux config into a yaml file
        job_name = "{}_{}.yml".format(meta, type_exp)
        with open(job_name, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)


if __name__ == "__main__":
    # Create (and optionally launch) the jobs!
    run(args)
