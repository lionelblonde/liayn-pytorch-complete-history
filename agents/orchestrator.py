import time
from copy import deepcopy
import os
from collections import namedtuple, deque, OrderedDict

import wandb
import numpy as np

from helpers import logger
from helpers.distributed_util import sync_check, mpi_mean_reduce
from agents.memory import RingBuffer
from helpers.console_util import timed_cm_wrapper, log_iter_info


def rollout_generator(env, agent, rollout_len):

    pixels = len(env.observation_space.shape) >= 3

    # Reset noise processes
    agent.reset_noise()

    t = 0
    done = True
    env_rew = 0.0
    ob = env.reset()

    if pixels:
        ob = np.array(ob)

    obs = RingBuffer(rollout_len, shape=agent.ob_shape)
    acs = RingBuffer(rollout_len, shape=agent.ac_shape)
    if hasattr(agent, 'disc'):
        syn_rews = RingBuffer(rollout_len, shape=(1,), dtype='float32')
    env_rews = RingBuffer(rollout_len, shape=(1,), dtype='float32')
    dones = RingBuffer(rollout_len, shape=(1,), dtype='int32')

    while True:
        ac = agent.predict(ob, apply_noise=True)

        # NaN-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        if t > 0 and t % rollout_len == 0:

            obs_ = obs.data.reshape(-1, *agent.ob_shape)

            if not pixels:
                agent.rms_obs.update(obs_)

            out = {"obs": obs_,
                   "acs": acs.data.reshape(-1, *agent.ac_shape),
                   "env_rews": env_rews.data.reshape(-1, 1),
                   "dones": dones.data.reshape(-1, 1)}
            if hasattr(agent, 'disc'):
                out.update({"syn_rews": syn_rews.data.reshape(-1, 1)})

            yield out

        obs.append(ob)
        acs.append(ac)
        dones.append(done)

        # Interact with env(s)
        new_ob, env_rew, done, _ = env.step(ac)

        env_rews.append(env_rew)

        if hasattr(agent, 'disc'):
            syn_rew = np.asscalar(agent.disc.get_reward(ob, ac).cpu().numpy().flatten())
            syn_rews.append(syn_rew)

        # Store transition(s) in the replay buffer
        if hasattr(agent, 'disc'):
            agent.store_transition(ob, ac, syn_rew, new_ob, done)
        else:
            agent.store_transition(ob, ac, env_rew, new_ob, done)

        ob = deepcopy(new_ob)
        if pixels:
            ob = np.array(ob)

        if done:
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


def evaluate(env,
             agent_wrapper,
             num_trajs,
             iter_num,
             render,
             model_path):

    # Rebuild the computational graph
    # Create an agent
    agent = agent_wrapper()
    # Create episode generator
    ep_gen = ep_generator(env, agent, render)
    # Initialize and load the previously learned weights into the freshly re-built graph

    # Load the model
    agent.load(model_path, iter_num)
    logger.info("model loaded from path:\n  {}".format(model_path))

    # Initialize the history data structures
    ep_lens = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, num_trajs))
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
          env,
          eval_env,
          agent_wrapper,
          experiment_name,
          ckpt_dir,
          save_frequency,
          pn_adapt_frequency,
          rollout_len,
          batch_size,
          training_steps_per_iter,
          eval_steps_per_iter,
          eval_frequency,
          actor_update_delay,
          d_update_ratio,
          render,
          record,
          expert_dataset,
          add_demos_to_mem,
          num_timesteps):

    assert training_steps_per_iter % actor_update_delay == 0, "must be a multiple"

    # Create an agent
    agent = agent_wrapper()

    if add_demos_to_mem:
        # Add demonstrations to memory
        agent.replay_buffer.add_demo_transitions_to_mem(expert_dataset)

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger)

    num_iters = num_timesteps // rollout_len
    iters_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()

    # Define rolling buffers for experiental data collection
    maxlen = 100
    keys = ['ac', 'actr_gradnorms', 'actr_losses', 'crit_gradnorms', 'crit_losses']
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        keys.extend(['eval_len', 'eval_env_ret'])
    if agent.hps.clipped_double:
        keys.extend(['twin_gradnorms', 'twin_losses'])
    if agent.param_noise is not None:
        keys.extend(['pn_dist', 'pn_cur_std'])
    Deques = namedtuple('Deques', keys)
    deques = Deques(**{k: deque(maxlen=maxlen) for k in keys})

    # Set up model save directory
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Setup wandb
    if rank == 0:
        wandb.init(project="DDPG-SAM",
                   name=experiment_name,
                   group='.'.join(experiment_name.split('.')[:-1]),
                   config=args.__dict__)

    # Create rollout generator for training the agent
    roll_gen = rollout_generator(env, agent, rollout_len)
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        # Create episode generator for evaluating the agent
        eval_ep_gen = ep_generator(eval_env, agent, render, record)

    while iters_so_far <= num_iters:

        log_iter_info(logger, iters_so_far, num_iters, tstart)

        if iters_so_far % 20 == 0:
            # Check if the mpi workers are still synced
            sync_check(agent.actr)
            sync_check(agent.crit)
            if agent.hps.clipped_double:
                sync_check(agent.twin)
            if hasattr(agent, 'disc'):
                sync_check(agent.disc)

        if rank == 0 and iters_so_far % save_frequency == 0:
            # Save the model
            agent.save(ckpt_dir, iters_so_far)
            logger.info("saving model @: {}".format(ckpt_dir))

        # Sample mini-batch in env w/ perturbed actor and store transitions
        with timed("interacting"):
            rollout = roll_gen.__next__()
            logger.info("[INFO] {} ".format("timesteps".ljust(20, '.')) +
                        "{}".format(timesteps_so_far + rollout_len))

        # Extend deques with collected experiential data
        deques.ac.extend(rollout['acs'])

        with timed("training"):
            for training_step in range(training_steps_per_iter):

                if agent.param_noise is not None:
                    if training_step % pn_adapt_frequency == 0:
                        # Adapt parameter noise
                        agent.adapt_param_noise()
                        # Store the action-space dist between perturbed and non-perturbed actors
                        deques.pn_dist.append(agent.pn_dist)
                        # Store the new std resulting from the adaption
                        deques.pn_cur_std.append(agent.param_noise.cur_std)

                # Train the actor-critic architecture
                update_critic = True
                if hasattr(agent, 'disc'):
                    update_critic = not bool(training_step % d_update_ratio)
                update_actor = update_critic and not bool(training_step % actor_update_delay)

                losses, gradnorms, lrnows = agent.train(update_critic=update_critic,
                                                        update_actor=update_actor,
                                                        iters_so_far=iters_so_far)
                # Store the losses and gradients in their respective deques
                deques.actr_gradnorms.append(gradnorms['actr'])
                deques.actr_losses.append(losses['actr'])
                deques.crit_gradnorms.append(gradnorms['crit'])
                deques.crit_losses.append(losses['crit'])
                if agent.hps.clipped_double:
                    deques.twin_gradnorms.append(gradnorms['twin'])
                    deques.twin_losses.append(losses['twin'])

            # Log statistics
            stats = OrderedDict()
            ac_np_mean = np.mean(deques.ac, axis=0)  # vector
            stats.update({'ac': {'min': np.amin(ac_np_mean),
                                 'max': np.amax(ac_np_mean),
                                 'mean': np.mean(ac_np_mean),
                                 'mpimean': mpi_mean_reduce(ac_np_mean)}})
            stats.update({'actr': {'loss': np.mean(deques.actr_losses),
                                   'gradnorm': np.mean(deques.actr_gradnorms),
                                   'lrnow': lrnows['actr'][0]}})
            stats.update({'crit': {'loss': np.mean(deques.crit_losses),
                                   'gradnorm': np.mean(deques.crit_gradnorms),
                                   'lrnow': lrnows['crit'][0]}})
            if agent.hps.clipped_double:
                stats.update({'twin': {'loss': np.mean(deques.twin_losses),
                                       'gradnorm': np.mean(deques.twin_gradnorms),
                                       'lrnow': lrnows['twin'][0]}})
            if agent.param_noise is not None:
                stats.update({'pn': {'pn_dist': np.mean(deques.pn_dist),
                                     'pn_cur_std': np.mean(deques.pn_cur_std)}})

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

            if iters_so_far % eval_frequency == 0:

                with timed("evaluating"):

                    # Use the running stats of the training environment to normalize
                    if hasattr(eval_env, 'running_moments'):
                        eval_env.running_moments = deepcopy(env.running_moments)

                    for eval_step in range(eval_steps_per_iter):
                        # Sample an episode w/ non-perturbed actor w/o storing anything
                        eval_ep = eval_ep_gen.__next__()
                        # Aggregate data collected during the evaluation to the buffers
                        deques.eval_len.append(eval_ep['ep_len'])
                        deques.eval_env_ret.append(eval_ep['ep_env_ret'])

                    # Log evaluation stats
                    logger.record_tabular('ep_len', np.mean(deques.eval_len))
                    logger.record_tabular('ep_env_ret', np.mean(deques.eval_env_ret))
                    logger.info("[CSV] dumping eval stats in .csv file")
                    logger.dump_tabular()

                    if record:
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

            if iters_so_far % eval_frequency == 0:
                wandb.log({'eval_len': np.mean(deques.eval_len),
                           'eval_env_ret': np.mean(deques.eval_env_ret)},
                          step=timesteps_so_far)
            if agent.param_noise is not None:
                wandb.log({'pn_dist': np.mean(deques.pn_dist),
                           'pn_cur_std': np.mean(deques.pn_cur_std)},
                          step=timesteps_so_far)
            wandb.log({'actr_loss': np.mean(deques.actr_losses),
                       'actr_gradnorm': np.mean(deques.actr_gradnorms),
                       'actr_lrnow': np.array(lrnows['actr']),
                       'crit_loss': np.mean(deques.crit_losses),
                       'crit_gradnorm': np.mean(deques.crit_gradnorms),
                       'crit_lrnow': np.array(lrnows['crit'])},
                      step=timesteps_so_far)
            if agent.hps.clipped_double:
                wandb.log({'twin_loss': np.mean(deques.twin_losses),
                           'twin_gradnorm': np.mean(deques.twin_gradnorms),
                           'twin_lrnow': np.array(lrnows['twin'])},
                          step=timesteps_so_far)

        iters_so_far += 1
        timesteps_so_far += rollout_len
