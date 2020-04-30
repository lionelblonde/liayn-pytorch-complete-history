import time
from copy import deepcopy
import os
import os.path as osp
from collections import defaultdict, deque

import wandb
import numpy as np

from helpers import logger
# from helpers.distributed_util import sync_check
from helpers.console_util import timed_cm_wrapper, log_iter_info
from agents.agent import Agent


def rollout_generator(env, agent, rollout_len):

    t = 0
    # Reset agent's noise process
    agent.reset_noise()
    # Reset agent's env
    ob = np.array(env.reset())

    while True:

        # Predict action
        ac = agent.predict(ob, apply_noise=True)
        # NaN-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        if t > 0 and t % rollout_len == 0:
            yield

        # Interact with env(s)
        new_ob, _, done, _ = env.step(ac)

        if agent.hps.wrap_absorb:
            _ob = np.append(ob, 0)
            _ac = np.append(ac, 0)
            if done and not env._elapsed_steps == env._max_episode_steps:
                # Wrap with an absorbing state
                _new_ob = np.append(np.zeros(agent.ob_shape), 1)
                _rew = agent.get_syn_rew(_ob[None], _ac[None], _new_ob[None])
                _rew = np.asscalar(_rew.cpu().numpy().flatten())
                transition = {
                    "obs0": _ob,
                    "acs": _ac,
                    "obs1": _new_ob,
                    "rews": _rew,
                    "dones1": done,
                    "obs0_orig": ob,
                    "acs_orig": ac,
                    "obs1_orig": new_ob,
                }
                agent.store_transition(transition)
                # Add absorbing transition
                _ob_a = np.append(np.zeros(agent.ob_shape), 1)
                _ac_a = np.append(np.zeros(agent.ac_shape), 1)
                _new_ob_a = np.append(np.zeros(agent.ob_shape), 1)
                _rew_a = agent.get_syn_rew(_ob_a[None], _ac_a[None], _new_ob_a[None])
                _rew_a = np.asscalar(_rew_a.cpu().numpy().flatten())
                transition_a = {
                    "obs0": _ob_a,
                    "acs": _ac_a,
                    "obs1": _new_ob_a,
                    "rews": _rew_a,
                    "dones1": done,
                    "obs0_orig": ob,  # from previous transition, with reward eval on absorbing
                    "acs_orig": ac,  # from previous transition, with reward eval on absorbing
                    "obs1_orig": new_ob,  # from previous transition, with reward eval on absorbing
                }
                agent.store_transition(transition_a)
            else:
                _new_ob = np.append(new_ob, 0)
                _rew = agent.get_syn_rew(_ob[None], _ac[None], _new_ob[None])
                _rew = np.asscalar(_rew.cpu().numpy().flatten())
                transition = {
                    "obs0": _ob,
                    "acs": _ac,
                    "obs1": _new_ob,
                    "rews": _rew,
                    "dones1": done,
                    "obs0_orig": ob,
                    "acs_orig": ac,
                    "obs1_orig": new_ob,
                }
                agent.store_transition(transition)
        else:
            rew = agent.get_syn_rew(ob[None], ac[None], new_ob[None])
            rew = np.asscalar(rew.cpu().numpy().flatten())
            transition = {
                "obs0": ob,
                "acs": ac,
                "obs1": new_ob,
                "rews": rew,
                "dones1": done,
            }
            agent.store_transition(transition)

        # Set current state with the next
        ob = np.array(deepcopy(new_ob))

        if done:
            # Reset agent's noise process
            agent.reset_noise()
            # Reset agent's env
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

    # Create collections
    d = defaultdict(list)
    b_eval = deque(maxlen=10)

    # Set up model save directory
    if rank == 0:
        ckpt_dir = osp.join(args.checkpoint_dir, experiment_name)
        os.makedirs(ckpt_dir, exist_ok=True)

    # Setup wandb
    if rank == 0:
        while True:
            try:
                wandb.init(project=args.wandb_project,
                           name=experiment_name,
                           group='.'.join(experiment_name.split('.')[:-2]),
                           job_type=experiment_name.split('.')[-2],
                           config=args.__dict__)
            except ConnectionRefusedError:
                pause = 5
                logger.info("wandb co error. Retrying in {} secs.".format(pause))
                time.sleep(pause)
                continue
            logger.info("wandb co established!")
            break

    # Create rollout generator for training the agent
    roll_gen = rollout_generator(env, agent, args.rollout_len)
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        # Create episode generator for evaluating the agent
        eval_ep_gen = ep_generator(eval_env, agent, args.render, args.record)

    while iters_so_far <= num_iters:

        log_iter_info(logger, iters_so_far, num_iters, tstart)

        # if iters_so_far % 20 == 0:
        #     # Check if the mpi workers are still synced
        #     sync_check(agent.actr)
        #     sync_check(agent.crit)
        #     if agent.hps.clipped_double:
        #         sync_check(agent.twin)
        #     sync_check(agent.disc)

        if rank == 0 and iters_so_far % args.save_frequency == 0:
            # Save the model
            agent.save(ckpt_dir, iters_so_far)
            logger.info("saving model @: {}".format(ckpt_dir))

        # Sample mini-batch in env with perturbed actor and store transitions
        with timed("interacting"):
            roll_gen.__next__()  # no need to get the returned rollout, stored in buffer

        with timed('training'):
            for training_step in range(args.training_steps_per_iter):

                if agent.param_noise is not None:
                    if training_step % args.pn_adapt_frequency == 0:
                        # Adapt parameter noise
                        agent.adapt_param_noise()
                        # Store the action-space dist between perturbed and non-perturbed actors
                        d['pn_dist'].append(agent.pn_dist)
                        # Store the new std resulting from the adaption
                        d['pn_cur_std'].append(agent.param_noise.cur_std)

                for _ in range(agent.hps.g_steps):
                    # Sample a batch of transitions from the replay buffer
                    batch = agent.sample_batch()
                    # Update the actor and critic
                    metrics, lrnows = agent.update_actor_critic(
                        batch=batch,
                        update_actor=not bool(iters_so_far % args.actor_update_delay),  # from TD3
                        iters_so_far=iters_so_far,
                    )
                    # Log training stats
                    d['actr_losses'].append(metrics['actr_loss'])
                    d['crit_losses'].append(metrics['crit_loss'])
                    if agent.hps.clipped_double:
                        d['twin_losses'].append(metrics['twin_loss'])
                    if agent.hps.prioritized_replay:
                        iws = metrics['iws']  # last one only
                    if agent.hps.kye_p and agent.hps.adaptive_aux_scaling:
                        d['cos_sims_p'].append(metrics['cos_sim_aux'])

                for _ in range(agent.hps.d_steps):
                    # Sample a batch of transitions from the replay buffer
                    batch = agent.sample_batch()
                    # Update the discriminator
                    metrics = agent.update_discriminator(batch)
                    # Log training stats
                    d['disc_losses'].append(metrics['disc_loss'])

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % args.eval_frequency == 0:

                with timed("evaluating"):
                    for eval_step in range(args.eval_steps_per_iter):
                        # Sample an episode w/ non-perturbed actor w/o storing anything
                        eval_ep = eval_ep_gen.__next__()
                        # Aggregate data collected during the evaluation to the buffers
                        d['eval_len'].append(eval_ep['ep_len'])
                        d['eval_env_ret'].append(eval_ep['ep_env_ret'])

                    b_eval.append(np.mean(d['eval_env_ret']))

        # Increment counters
        iters_so_far += 1
        timesteps_so_far += args.rollout_len

        if rank == 0:

            # Log stats in csv
            if (iters_so_far - 1) % args.eval_frequency == 0:
                logger.record_tabular('timestep', timesteps_so_far)
                logger.record_tabular('eval_len', np.mean(d['eval_len']))
                logger.record_tabular('eval_env_ret', np.mean(d['eval_env_ret']))
                logger.record_tabular('avg_eval_env_ret', np.mean(b_eval))
                if agent.hps.kye_p and agent.hps.adaptive_aux_scaling:
                    logger.record_tabular('cos_sim_p', np.mean(d['cos_sims_p']))
                logger.info("dumping stats in .csv file")
                logger.dump_tabular()

            if ((iters_so_far - 1) % args.eval_frequency == 0) and args.record:
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
            wandb.log({"num_workers": np.array(world_size)})
            if agent.hps.prioritized_replay:
                quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                np.quantile(iws, quantiles)
                wandb.log({"q{}".format(q): np.quantile(iws, q)
                           for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
                          step=timesteps_so_far)
            if agent.param_noise is not None:
                wandb.log({'pn_dist': np.mean(d['pn_dist']),
                           'pn_cur_std': np.mean(d['pn_cur_std'])},
                          step=timesteps_so_far)
            wandb.log({'actr_loss': np.mean(d['actr_losses']),
                       'actr_lrnow': np.array(lrnows['actr']),
                       'crit_loss': np.mean(d['crit_losses']),
                       'crit_lrnow': np.array(lrnows['crit']),
                       'disc_loss': np.mean(d['disc_losses'])},
                      step=timesteps_so_far)
            if agent.hps.clipped_double:
                wandb.log({'twin_loss': np.mean(d['twin_losses']),
                           'twin_lrnow': np.array(lrnows['twin'])},
                          step=timesteps_so_far)
            if agent.hps.kye_p and agent.hps.adaptive_aux_scaling:
                wandb.log({'cos_sim_p': np.mean(d['cos_sims_p'])},
                          step=timesteps_so_far)

            if (iters_so_far - 1) % args.eval_frequency == 0:
                wandb.log({'eval_len': np.mean(d['eval_len']),
                           'eval_env_ret': np.mean(d['eval_env_ret']),
                           'avg_eval_env_ret': np.mean(b_eval)},
                          step=timesteps_so_far)

        # Clear the iteration's running stats
        d.clear()
