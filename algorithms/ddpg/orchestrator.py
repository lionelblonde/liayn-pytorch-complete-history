import time
import copy
import os
import os.path as osp
from collections import namedtuple, deque, OrderedDict
import yaml

import numpy as np
import visdom

from algorithms.helpers import logger
from algorithms.ddpg.memory import RingBuffer
from algorithms.helpers.console_util import (timed_cm_wrapper, pretty_iter,
                                             pretty_elapsed, columnize)


def rollout_generator(env, agent, rollout_len, prefill=0):

    # Reset noise processes
    agent.reset_noise()

    t = 0
    done = True
    env_rew = 0.0
    ob = env.reset()

    obs = RingBuffer(rollout_len, shape=(agent.ob_dim,))
    acs = RingBuffer(rollout_len, shape=(agent.ac_dim,))
    qs = RingBuffer(rollout_len, shape=(1,), dtype='float32')
    env_rews = RingBuffer(rollout_len, shape=(1,), dtype='float32')
    dones = RingBuffer(rollout_len, shape=(1,), dtype='int32')

    while True:
        ac, q_pred = agent.predict(ob, apply_noise=True)
        if t < prefill:
            logger.info("populating the replay buffer with uniform policy")
            # Override predicted action with actions which are sampled
            # from a uniform random distribution over valid actions
            ac = env.action_space.sample()

        if t > 0 and t % rollout_len == 0:
            yield {"obs": obs.data.reshape(-1, agent.ob_dim),
                   "acs": acs.data.reshape(-1, agent.ac_dim),
                   "qs": qs.data.reshape(-1, 1),
                   "env_rews": env_rews.data.reshape(-1, 1),
                   "dones": dones.data.reshape(-1, 1)}

            _, q_pred = agent.predict(ob, apply_noise=True)

        obs.append(ob)
        acs.append(ac)
        qs.append(q_pred)
        dones.append(done)

        # ac = ac.reshape(-1, agent.ac_dim)
        # Interact with env(s)
        new_ob, env_rew, done, _ = env.step(ac)

        # env_rew = env_rew.reshape(-1, 1)
        env_rews.append(env_rew)
        # done = done.reshape(-1, 1)

        # Store transition(s) in the replay buffer
        agent.store_transition(ob, ac, env_rew, new_ob, done)

        assert isinstance(ob, np.ndarray), "copy ignored for torch tensors -> clone"
        ob = copy.copy(new_ob)

        if done:
            agent.reset_noise()
            ob = env.reset()

        t += 1


def ep_generator(env, agent, render):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """

    ob = env.reset()
    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    acs = []
    qs = []
    env_rews = []

    while True:
        ac, q = agent.predict(ob, apply_noise=False)
        obs.append(ob)
        acs.append(ac)
        qs.append(q)
        new_ob, env_rew, done, _ = env.step(ac)
        if render:
            env.render()
        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        ob = copy.copy(new_ob)
        if done:
            obs = np.array(obs)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            yield {"obs": obs,
                   "acs": acs,
                   "qs": qs,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_env_ret": cur_ep_env_ret}
            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            acs = []
            env_rews = []
            agent.reset_noise()
            ob = env.reset()


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
          enable_visdom,
          visdom_dir,
          save_frequency,
          pn_adapt_frequency,
          rollout_len,
          batch_size,
          training_steps_per_iter,
          eval_steps_per_iter,
          eval_frequency,
          actor_update_delay,
          render,
          expert_dataset,
          add_demos_to_mem,
          prefill,
          max_iters):

    assert not training_steps_per_iter % actor_update_delay, "must be a multiple"

    # Create an agent
    agent = agent_wrapper()

    if add_demos_to_mem:
        # Add demonstrations to memory
        agent.replay_buffer.add_demo_transitions_to_mem(expert_dataset)

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger)

    # Create rollout generator for training the agent
    seg_gen = rollout_generator(env, agent, rollout_len, prefill)
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        # Create episode generator for evaluating the agent
        eval_ep_gen = ep_generator(eval_env, agent, render)

    iters_so_far = 0
    tstart = time.time()

    # Define rolling buffers for experiental data collection
    maxlen = 100
    keys = ['ac', 'q', 'actor_gradnorms', 'actor_losses', 'critic_gradnorms', 'critic_losses']
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        keys.extend(['eval_ac', 'eval_q', 'eval_len', 'eval_env_ret'])
    if agent.hps.enable_clipped_double:
        keys.extend(['twin_critic_gradnorms', 'twin_critic_losses'])
    if agent.param_noise is not None:
        keys.extend(['pn_dist', 'pn_cur_std'])
    Deques = namedtuple('Deques', keys)
    deques = Deques(**{k: deque(maxlen=maxlen) for k in keys})

    # Set up model save directory
    assert not osp.exists(ckpt_dir)
    os.makedirs(ckpt_dir)
    path = osp.join(ckpt_dir, experiment_name)

    # Setup Visdom
    if rank == 0 and enable_visdom:

        # Make a copy of hps for visdom
        viz = visdom.Visdom(env="job_{}".format(experiment_name),
                            log_to_filename=visdom_dir)
        assert viz.check_connection(timeout_seconds=4), "viz co not great"
        viz.text("World size: {}".format(world_size))
        iter_win = viz.text("will be overridden soon")
        mem_win = viz.text("will be overridden soon")
        viz.text(yaml.dump(args.__dict__, default_flow_style=False))

        keys = ['eval_len', 'eval_env_ret']
        if agent.param_noise is not None:
            keys.extend(['pn_dist', 'pn_cur_std'])

        keys.extend(['actor_loss', 'critic_loss'])
        if agent.hps.enable_clipped_double:
            keys.extend(['twin_critic_loss'])

        # Create (empty) visdom windows
        VizWins = namedtuple('VizWins', keys)
        vizwins = VizWins(**{k: viz.line(X=[0], Y=[np.nan]) for k in keys})
        # HAXX: NaNs ignored by visdom

    while iters_so_far <= max_iters:

        pretty_iter(logger, iters_so_far)
        pretty_elapsed(logger, tstart)

        # Save the model
        if iters_so_far % save_frequency == 0:
            agent.save(path, iters_so_far)
            logger.info("saving model:\n  @: {}".format(path))

        # Sample mini-batch in env w/ perturbed actor and store transitions
        with timed("interacting"):
            seg = seg_gen.__next__()

        # Extend deques with collected experiential data
        deques.ac.extend(seg['acs'])
        deques.q.extend(seg['qs'])

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
                update_actor = not bool(training_step % actor_update_delay)
                losses, gradnorms = agent.train(update_actor=update_actor)
                # Store the losses and gradients in their respective deques
                deques.actor_gradnorms.append(gradnorms['actor'])
                deques.actor_losses.append(losses['actor'])
                deques.critic_gradnorms.append(gradnorms['critic'])
                deques.critic_losses.append(losses['critic'])
                if agent.hps.enable_clipped_double:
                    deques.twin_critic_gradnorms.append(gradnorms['twin_critic'])
                    deques.twin_critic_losses.append(losses['twin_critic'])

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % eval_frequency == 0:

                with timed("evaluating"):
                    for eval_step in range(eval_steps_per_iter):

                        # Sample an episode w/ non-perturbed actor w/o storing anything
                        eval_ep = eval_ep_gen.__next__()

                        # Aggregate data collected during the evaluation to the buffers
                        deques.eval_ac.extend(eval_ep['acs'])
                        deques.eval_q.extend(eval_ep['qs'])
                        deques.eval_len.append(eval_ep['ep_len'])
                        deques.eval_env_ret.append(eval_ep['ep_env_ret'])

        # Log statistics

        logger.info("logging misc training stats")

        stats = OrderedDict()

        # Add min, max and mean of the components of the average action
        ac_np_mean = np.mean(deques.ac, axis=0)  # vector
        stats.update({'min_ac_comp': np.amin(ac_np_mean)})
        stats.update({'max_ac_comp': np.amax(ac_np_mean)})
        stats.update({'mean_ac_comp': np.mean(ac_np_mean)})

        # Add Q values mean and std
        stats.update({'q_value': np.mean(deques.q)})
        stats.update({'q_deviation': np.std(deques.q)})

        # Add gradient norms
        stats.update({'actor_gradnorm': np.mean(deques.actor_gradnorms)})
        stats.update({'critic_gradnorm': np.mean(deques.critic_gradnorms)})

        if agent.hps.enable_clipped_double:
            stats.update({'twin_critic_gradnorm': np.mean(deques.critic_gradnorms)})

        if agent.param_noise is not None:
            stats.update({'pn_dist': np.mean(deques.pn_dist)})
            stats.update({'pn_cur_std': np.mean(deques.pn_cur_std)})

        # Add replay buffer num entries
        stats.update({'mem_num_entries': agent.replay_buffer.num_entries})

        # Log dictionary content
        logger.info(columnize(['name', 'value'], stats.items(), [24, 16]))

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % eval_frequency == 0:

                # Use the logger object to log the eval stats (will appear in `progress{}.csv`)
                logger.info("logging misc eval stats")
                # Add min, max and mean of the components of the average action
                ac_np_mean = np.mean(deques.eval_ac, axis=0)  # vector
                logger.record_tabular('min_ac_comp', np.amin(ac_np_mean))
                logger.record_tabular('max_ac_comp', np.amax(ac_np_mean))
                logger.record_tabular('mean_ac_comp', np.mean(ac_np_mean))
                # Add Q values mean and std
                logger.record_tabular('q_value', np.mean(deques.eval_q))
                logger.record_tabular('q_deviation', np.std(deques.eval_q))
                # Add episodic stats
                logger.record_tabular('ep_len', np.mean(deques.eval_len))
                logger.record_tabular('ep_env_ret', np.mean(deques.eval_env_ret))
                logger.dump_tabular()

        # Mark the end of the iter in the logs
        logger.info('')

        iters_so_far += 1

        if rank == 0 and enable_visdom:

            viz.text("Current iter: {}".format(iters_so_far), win=iter_win, append=False)
            filled_ratio = agent.replay_buffer.num_entries / (1. * args.mem_size)  # HAXX
            viz.text("Replay buffer: {} (filled ratio: {})".format(agent.replay_buffer.num_entries,
                                                                   filled_ratio),
                     win=mem_win,
                     append=False)

            if iters_so_far % eval_frequency == 0:
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.eval_len)],
                         win=vizwins.eval_len,
                         update='append',
                         opts=dict(title='Eval Episode Length'))
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.eval_env_ret)],
                         win=vizwins.eval_env_ret,
                         update='append',
                         opts=dict(title='Eval Episodic Return'))

            if agent.param_noise is not None:
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.pn_dist)],
                         win=vizwins.pn_dist,
                         update='append',
                         opts=dict(title='Distance in action space (param noise)'))
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.pn_cur_std)],
                         win=vizwins.pn_cur_std,
                         update='append',
                         opts=dict(title='Parameter-Noise Current Std Dev'))

            viz.line(X=[iters_so_far],
                     Y=[np.mean(deques.actor_losses)],
                     win=vizwins.actor_loss,
                     update='append',
                     opts=dict(title="Actor Loss"))

            viz.line(X=[iters_so_far],
                     Y=[np.mean(deques.critic_losses)],
                     win=vizwins.critic_loss,
                     update='append',
                     opts=dict(title="Critic Loss"))

            if agent.hps.enable_clipped_double:
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.twin_critic_losses)],
                         win=vizwins.twin_critic_loss,
                         update='append',
                         opts=dict(title="Twin Critic Loss"))
