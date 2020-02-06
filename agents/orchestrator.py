import time
from copy import deepcopy
import os
import os.path as osp
from collections import defaultdict, OrderedDict

import wandb
import numpy as np

from helpers import logger
from helpers.distributed_util import sync_check, mpi_mean_reduce
from helpers.console_util import timed_cm_wrapper, log_iter_info
from agents.agent import Agent


def rollout_generator(env, agent, rollout_len):

    t = 0
    rollout = defaultdict(list)

    if not agent.hps.pixels:
        # Warm-start running stats with expert observations
        agent.rms_obs.update(agent.expert_dataset.data['obs0'])

    # Reset agent's noise processes and env
    agent.reset_noise()
    ob = np.array(env.reset())

    while True:

        # Predict action
        ac = agent.predict(ob, apply_noise=True)
        # NaN-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        if t > 0 and t % rollout_len == 0:
            obs = np.array(rollout["obs"]).reshape(-1, *agent.ob_shape)
            out = {
                "obs": obs,
                "acs": np.array(rollout["acs"]).reshape(-1, *agent.ac_shape),
            }
            if not agent.hps.pixels:
                # Update running stats
                agent.rms_obs.update(obs)
            # Yield
            yield out
            # When going back in, clear the rollout
            rollout.clear()

        # Interact with env(s)
        new_ob, _, done, _ = env.step(ac)

        # Store transition(s) in the replay buffer
        rew = np.asscalar(agent.get_reward(ob, ac).cpu().numpy().flatten())
        transition = {"obs0": ob,
                      "acs": ac,
                      "rews": rew,
                      "obs1": new_ob,
                      "dones1": done}
        agent.replay_buffer.append(transition)

        # Populate rollout
        rollout["obs"].append(ob)
        rollout["acs"].append(ac)

        # Set current state with the next
        ob = np.array(deepcopy(new_ob))

        if done:
            # Reset agent's noise processes and env
            agent.reset_noise()
            ob = np.array(env.reset())

        t += 1


def ep_generator(env, agent, render, record):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """

    kwargs = {'mode': 'rgb_array'}

    def _render():
        return env.render(**kwargs)

    ob = np.array(env.reset())

    if record:
        ob_orig = _render()

    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    if record:
        obs_render = []
    acs = []
    env_rews = []

    while True:

        # Predict action
        ac = agent.predict(ob, apply_noise=False)
        # NaN-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        obs.append(ob)
        if record:
            obs_render.append(ob_orig)
        acs.append(ac)
        new_ob, env_rew, done, _ = env.step(ac)

        if render:
            env.render()

        if record:
            ob_orig = _render()

        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        ob = np.array(deepcopy(new_ob))
        if done:
            obs = np.array(obs)
            if record:
                obs_render = np.array(obs_render)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            out = {"obs": obs,
                   "acs": acs,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_env_ret": cur_ep_env_ret}
            if record:
                out.update({"obs_render": obs_render})

            yield out

            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            if record:
                obs_render = []
            acs = []
            env_rews = []
            agent.reset_noise()
            ob = np.array(env.reset())

            if record:
                ob_orig = _render()


def evaluate(args, device, env):

    # Rebuild the computational graph
    # Create an agent
    agent = Agent(env=env,
                  device=device,
                  hps=args,
                  expert_dataset=None)
    # Create episode generator
    ep_gen = ep_generator(env, agent, args.render)
    # Initialize and load the previously learned weights into the freshly re-built graph

    # Load the model
    agent.load(args.model_path, args.iter_num)
    logger.info("model loaded from path:\n  {}".format(args.model_path))

    # Initialize the history data structures
    ep_lens = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(args.num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, args.num_trajs))
        traj = ep_gen.__next__()
        ep_len, ep_env_ret = traj['ep_len'], traj['ep_env_ret']
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_env_rets.append(ep_env_ret)
    # Log some statistics of the collected trajectories
    ep_len_mean = np.mean(ep_lens)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()


def learn(args,
          rank,
          world_size,
          device,
          env,
          eval_env,
          experiment_name,
          expert_dataset):

    assert args.training_steps_per_iter % args.actor_update_delay == 0, "must be a multiple"

    # Create an agent
    agent = Agent(env=env,
                  device=device,
                  hps=args,
                  expert_dataset=expert_dataset)

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger)

    # Start clocks
    num_iters = int(args.num_timesteps) // args.rollout_len
    iters_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()

    # Create dictionary to collect stats
    d = defaultdict(list)

    # Set up model save directory
    if rank == 0:
        ckpt_dir = osp.join(args.checkpoint_dir, experiment_name)
        os.makedirs(ckpt_dir, exist_ok=True)

    # Setup wandb
    if rank == 0:
        wandb.init(project=args.wandb_project,
                   name=experiment_name,
                   group='.'.join(experiment_name.split('.')[:-2]),
                   job_type=experiment_name.split('.')[-2],
                   config=args.__dict__)

    # Create rollout generator for training the agent
    roll_gen = rollout_generator(env, agent, args.rollout_len)
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        # Create episode generator for evaluating the agent
        eval_ep_gen = ep_generator(eval_env, agent, args.render, args.record)

    while iters_so_far <= num_iters:

        log_iter_info(logger, iters_so_far, num_iters, tstart)

        if iters_so_far % 20 == 0:
            # Check if the mpi workers are still synced
            sync_check(agent.actr)
            sync_check(agent.crit)
            if agent.hps.clipped_double:
                sync_check(agent.twin)
            sync_check(agent.disc)

        if rank == 0 and iters_so_far % args.save_frequency == 0:
            # Save the model
            agent.save(ckpt_dir, iters_so_far)
            logger.info("saving model @: {}".format(ckpt_dir))

        # Sample mini-batch in env with perturbed actor and store transitions
        with timed("interacting"):
            rollout = roll_gen.__next__()
            logger.info("[INFO] {} ".format("timesteps".ljust(20, '.')) +
                        "{}".format(timesteps_so_far + args.rollout_len))

        with timed("training"):
            for training_step in range(args.training_steps_per_iter):

                if agent.param_noise is not None:
                    if training_step % args.pn_adapt_frequency == 0:
                        # Adapt parameter noise
                        agent.adapt_param_noise()
                        # Store the action-space dist between perturbed and non-perturbed actors
                        d['pn_dist'].append(agent.pn_dist)
                        # Store the new std resulting from the adaption
                        d['pn_cur_std'].append(agent.param_noise.cur_std)

                # Train the actor-critic architecture
                update_critic = True
                update_critic = not bool(training_step % args.d_update_ratio)
                update_actor = update_critic and not bool(training_step % args.actor_update_delay)
                losses, gradns, lrnows = agent.train(update_critic=update_critic,
                                                     update_actor=update_actor,
                                                     rollout=rollout,
                                                     iters_so_far=iters_so_far)
                d['actr_gradns'].append(gradns['actr'])
                d['actr_losses'].append(losses['actr'])
                d['crit_losses'].append(losses['crit'])
                if agent.hps.clipped_double:
                    d['twin_losses'].append(losses['twin'])

            # Log statistics
            stats = OrderedDict()
            ac_np_mean = np.mean(rollout['acs'], axis=0)  # vector
            stats.update({'ac': {'min': np.amin(ac_np_mean),
                                 'max': np.amax(ac_np_mean),
                                 'mean': np.mean(ac_np_mean),
                                 'mpimean': mpi_mean_reduce(ac_np_mean)}})
            stats.update({'actr': {'loss': np.mean(d['actr_losses']),
                                   'gradn': np.mean(d['actr_gradns']),
                                   'lrnow': lrnows['actr'][0]}})
            stats.update({'crit': {'loss': np.mean(d['crit_losses']),
                                   'lrnow': lrnows['crit'][0]}})
            if agent.hps.clipped_double:
                stats.update({'twin': {'loss': np.mean(d['twin_losses']),
                                       'lrnow': lrnows['twin'][0]}})
            if agent.param_noise is not None:
                stats.update({'pn': {'pn_dist': np.mean(d['pn_dist']),
                                     'pn_cur_std': np.mean(d['pn_cur_std'])}})

            num_entries = deepcopy(agent.replay_buffer.num_entries)
            stats.update({'memory': {'num_entries': str(num_entries),
                                     'capacity': str(agent.hps.mem_size),
                                     'load': "{:.2%}".format(num_entries /
                                                             agent.hps.mem_size)}})
            for k, v in stats.items():
                assert isinstance(v, dict)
                v_ = {a: "{:.5f}".format(b) if not isinstance(b, str) else b for a, b in v.items()}
                logger.info("[INFO] {} {}".format(k.ljust(20, '.'), v_))

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % args.eval_frequency == 0:

                with timed("evaluating"):

                    # Use the running stats of the training environment to normalize
                    if hasattr(eval_env, 'running_moments'):
                        eval_env.running_moments = deepcopy(env.running_moments)

                    for eval_step in range(args.eval_steps_per_iter):
                        # Sample an episode w/ non-perturbed actor w/o storing anything
                        eval_ep = eval_ep_gen.__next__()
                        # Aggregate data collected during the evaluation to the buffers
                        d['eval_len'].append(eval_ep['ep_len'])
                        d['eval_env_ret'].append(eval_ep['ep_env_ret'])

                    # Log evaluation stats
                    logger.record_tabular('ep_len', np.mean(d['eval_len']))
                    logger.record_tabular('ep_env_ret', np.mean(d['eval_env_ret']))
                    logger.info("[CSV] dumping eval stats in .csv file")
                    logger.dump_tabular()

                    if args.record:
                        # Record the last episode in a video
                        frames = np.split(eval_ep['obs_render'], 1, axis=-1)
                        frames = np.concatenate(np.array(frames), axis=0)
                        frames = np.array([np.squeeze(a, axis=0)
                                           for a in np.split(frames, frames.shape[0], axis=0)])
                        frames = np.transpose(frames, (0, 3, 1, 2))  # from nwhc to ncwh

                        wandb.log({'video': wandb.Video(frames.astype(np.uint8),
                                                        fps=25,
                                                        format='gif',
                                                        caption="Evaluation (last episode)")},
                                  step=timesteps_so_far)

        # Log stats in dashboard
        if rank == 0:

            wandb.log({"num_workers": np.array(world_size)},
                      step=timesteps_so_far)
            if iters_so_far % args.eval_frequency == 0:
                wandb.log({'eval_len': np.mean(d['eval_len']),
                           'eval_env_ret': np.mean(d['eval_env_ret'])},
                          step=timesteps_so_far)
            if agent.param_noise is not None:
                wandb.log({'pn_dist': np.mean(d['pn_dist']),
                           'pn_cur_std': np.mean(d['pn_cur_std'])},
                          step=timesteps_so_far)
            wandb.log({'actr_loss': np.mean(d['actr_losses']),
                       'actr_gradn': np.mean(d['actr_gradns']),
                       'actr_lrnow': np.array(lrnows['actr']),
                       'crit_loss': np.mean(d['crit_losses']),
                       'crit_lrnow': np.array(lrnows['crit'])},
                      step=timesteps_so_far)
            if agent.hps.clipped_double:
                wandb.log({'twin_loss': np.mean(d['twin_losses']),
                           'twin_lrnow': np.array(lrnows['twin'])},
                          step=timesteps_so_far)

        # Increment counters
        iters_so_far += 1
        timesteps_so_far += args.rollout_len
        # Clear the iteration's running stats
        d.clear()
