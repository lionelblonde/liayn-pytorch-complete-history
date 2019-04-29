import os
import os.path as osp
import random

import numpy as np
import torch

from algorithms.helpers import logger
from algorithms.helpers.argparsers import ddpg_argparser
from algorithms.helpers.experiment_initializer import ExperimentInitializer
from algorithms.helpers.env_makers import make_env
from algorithms.agents import orchestrator
from algorithms.agents.ddpg_agent import DDPGAgent
from algorithms.agents.evade_agent import EvadeAgent


def train(args):
    """Train an agent"""

    # Get the current process rank
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    # Override batch size
    args.batch_size = args.batch_size // world_size

    torch.set_num_threads(1)

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args, rank=rank, world_size=world_size)
    experiment.configure_logging()
    # Create experiment name
    experiment_name = experiment.get_long_name()

    # Set device-related knobs
    if args.cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if args.cuda else "cpu")
    logger.info("device in use: {}".format(device))

    # Seedify
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Override the specified seed
    args.seed += (1000000 * rank)

    # Create environment
    env = make_env(args.env_id, args.seed)

    expert_dataset = None
    if args.add_demos_to_mem or args.algo == 'evade':
        # Create the expert demonstrations dataset from expert trajectories
        from algorithms.agents.demo_dataset import DemoDataset
        expert_dataset = DemoDataset(expert_arxiv=args.expert_path, size=args.num_demos,
                                     train_fraction=None, randomize=True, full=True)

    # Create an agent wrapper
    if args.algo == 'ddpg':
        def agent_wrapper():
            return DDPGAgent(env=env, device=device, hps=args, comm=comm)
    elif args.algo == 'evade':
        def agent_wrapper():
            return EvadeAgent(env=env, device=device, hps=args, comm=comm,
                              expert_dataset=expert_dataset)
    else:
        raise NotImplementedError("algorithm not covered")

    # Create an evaluation environment not to mess up with training rollouts
    eval_env = None
    if rank == 0:
        eval_env = make_env(args.env_id, args.seed)

    # Train
    orchestrator.learn(args=args,
                       rank=rank,
                       world_size=world_size,
                       env=env,
                       eval_env=eval_env,
                       agent_wrapper=agent_wrapper,
                       experiment_name=experiment_name,
                       ckpt_dir=osp.join(args.checkpoint_dir, experiment_name),
                       enable_visdom=args.enable_visdom,
                       visdom_dir=osp.join(args.visdom_dir, experiment_name),
                       save_frequency=args.save_frequency,
                       pn_adapt_frequency=args.pn_adapt_frequency,
                       rollout_len=args.rollout_len,
                       batch_size=args.batch_size,
                       training_steps_per_iter=args.training_steps_per_iter,
                       eval_steps_per_iter=args.eval_steps_per_iter,
                       eval_frequency=args.eval_frequency,
                       actor_update_delay=args.actor_update_delay,
                       d_update_ratio=args.d_update_ratio,
                       render=args.render,
                       expert_dataset=expert_dataset,
                       add_demos_to_mem=args.add_demos_to_mem,
                       prefill=args.prefill,
                       max_iters=int(args.num_iters))

    # Close environment
    env.close()

    # Close the eval env
    if eval_env is not None:
        assert rank == 0
        eval_env.close()


def evaluate(args):
    """Evaluate an agent"""

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()

    # Seedify
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create environment
    env = make_env(args.env_id, args.seed)

    if args.record:
        # Create experiment name
        experiment_name = experiment.get_long_name()
        save_dir = osp.join(args.video_dir, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        # Wrap the environment again to record videos
        from algorithms.helpers.video_recorder import VideoRecorder
        env = VideoRecorder(env=env,
                            save_dir=save_dir,
                            record_video_trigger=lambda x: x % x == 0,  # record at the very start
                            video_length=args.video_len,
                            prefix="video_{}".format(args.env_id))

    # Create an agent wrapper
    if args.algo == 'ddpg':
        def agent_wrapper():
            return DDPGAgent(env=env, device='cpu', hps=args, comm=None)
    elif args.algo == 'evade':
        def agent_wrapper():
            return EvadeAgent(env=env, device='cpu', hps=args, comm=None)
    else:
        raise NotImplementedError("algorithm not covered")

    # Evaluate agent trained via DDPG
    orchestrator.evaluate(env=env,
                          agent_wrapper=agent_wrapper,
                          num_trajs=args.num_trajs,
                          iter_num=args.iter_num,
                          render=args.render,
                          model_path=args.model_path)

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = ddpg_argparser().parse_args()
    if _args.task == 'train':
        train(_args)
    elif _args.task == 'evaluate':
        evaluate(_args)
    else:
        raise NotImplementedError
