from algorithms.helpers.misc_util import boolean_flag


def argparse(description):
    """Create an empty argparse.ArgumentParser"""
    import argparse
    return argparse.ArgumentParser(description=description,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def ddpg_argparser(description="DDPG Experiment"):
    """Create an argparse.ArgumentParser for DDPG-related tasks"""
    parser = argparse(description)
    parser.add_argument('--note', help='w/e note', type=str, default=None)
    parser.add_argument('--num_sbires', type=int, default=0,
                        help='number of parallel data collectors')
    boolean_flag(parser, 'cuda', default=False)
    parser.add_argument('--env_id', help='environment identifier', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='directory to save the models',
                        default='data/checkpoints')
    parser.add_argument('--log_dir', help='directory to save the log files',
                        default='data/logs')
    boolean_flag(parser, 'enable_visdom', help='whether to log to visdom', default=False)
    parser.add_argument('--visdom_dir', help='directory to save the visdom logs',
                        default='data/visdom')
    boolean_flag(parser, 'render', help='whether to render the interaction traces', default=False)
    boolean_flag(parser, 'record', help='whether to record the interaction traces', default=False)
    parser.add_argument('--video_dir', help='directory to save the video recordings',
                        default='data/videos')
    parser.add_argument('--video_len', help='duration of the video to record',
                        type=int, default=200)
    parser.add_argument('--task', type=str, choices=['train', 'evaluate'], default=None)
    parser.add_argument('--algo', type=str, choices=['ddpg', 'ad2d'], default=None)
    parser.add_argument('--save_frequency', help='save model every xx iterations',
                        type=int, default=100)
    parser.add_argument('--num_iters', help='cummulative number of iterations since launch',
                        type=int, default=int(1e6))
    parser.add_argument('--rollout_len', help='number of interactions per iteration',
                        type=int, default=16)
    parser.add_argument('--batch_size', help='minibatch size', type=int, default=32)
    parser.add_argument('--num_trajs', help='number of trajectories to evaluate',
                        type=int, default=10)
    parser.add_argument('--iter_num', help='iteration to evaluate the model at',
                        type=int, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--quantile_emb_dim', type=int, default=64, help='n in IQN paper')
    parser.add_argument('--num_tau', type=int, default=32, help='N in IQN paper')
    parser.add_argument('--num_tau_prime', type=int, default=32, help='N prime in IQN paper')
    parser.add_argument('--num_tau_tilde', type=int, default=32, help='K in IQN paper')
    boolean_flag(parser, 'with_layernorm', default=False)
    boolean_flag(parser, 'enable_clipped_double', default=True,
                 help='whether to use clipped double q learning')
    boolean_flag(parser, 'enable_targ_actor_smoothing', default=True,
                 help='whether to use target policy smoothing')
    parser.add_argument('--td3_std', type=float, default=0.2,
                        help='std of amoothing noise applied to the target action')
    parser.add_argument('--c', type=float, default=0.5,
                        help='limit for absolute value of target action smoothing noise')
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.995)
    parser.add_argument('--mem_size', type=int, default=int(1e6))
    boolean_flag(parser, 'prioritized_replay', default=False)
    parser.add_argument('--alpha', help='how much prioritized', type=float, default=0.3)
    boolean_flag(parser, 'ranked', default=False)
    boolean_flag(parser, 'unreal', default=False)
    parser.add_argument('--beta', help='importance weights usage', default=1.0, type=float)
    parser.add_argument('--reward_scale', type=float, default=1.)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--noise_type', help='choices: adaptive-param_xx, normal_xx, ou_xx, none',
                        type=str, default='adaptive-param_0.2, ou_0.1, normal_0.1')
    parser.add_argument('--pn_adapt_frequency', type=float, default=50)
    parser.add_argument('--polyak', type=float, default=0.005, help='target networks tracking')
    parser.add_argument('--wd_scale', help='critic wd scale', type=float, default=0.001)
    boolean_flag(parser, 'n_step_returns', default=True)
    parser.add_argument('--n', help='number of steps for the TD lookahead', type=int, default=10)
    parser.add_argument('--training_steps_per_iter', type=int, default=50)
    parser.add_argument('--eval_steps_per_iter', type=int, default=100)
    parser.add_argument('--eval_frequency', type=int, default=500)
    parser.add_argument('--prefill', type=int, default=0,
                        help='number of initial steps during which actions are uniformly picked')
    parser.add_argument('--actor_update_delay', type=int, default=1,
                        help='number of critic updates to perform per actor update')
    boolean_flag(parser, 'add_demos_to_mem', default=False)
    parser.add_argument('--expert_path', help='.npz archive containing the demos',
                        type=str, default=None)
    parser.add_argument('--num_demos', help='number of expert demo trajs for imitation',
                        type=int, default=None)
    return parser
