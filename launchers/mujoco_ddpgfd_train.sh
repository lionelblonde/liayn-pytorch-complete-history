#!/usr/bin/env bash
# Example: ./mujoco_ddpgfd_train.sh <env_id> <num_learners> <num_sbires> <expert_path> <num_demos>

cd ..

mpirun -np $2 python -m algorithms.ddpg.run \
    --no-cuda \
    --env_id=$1 \
    --seed=0 \
    --checkpoint_dir="data/checkpoints" \
    --no-enable_visdom \
    --visdom_dir="data/visdom" \
    --log_dir="data/logs" \
    --task="train" \
    --algo="ddpg" \
    --save_frequency=100 \
    --num_iters=10000000 \
    --training_steps_per_iter=20 \
    --eval_steps_per_iter=20 \
    --eval_frequency=10 \
    --prefill=200 \
    --no-render \
    --rollout_len=5 \
    --batch_size=128 \
    --polyak=0.005 \
    --with_layernorm \
    --reward_scale=1. \
    --no-enable_clipped_double \
    --no-enable_targ_actor_smoothing \
    --actor_update_delay=2 \
    --actor_lr=1e-3 \
    --critic_lr=1e-3 \
    --clip_norm=40. \
    --noise_type="adaptive-param_0.2, ou_0.2" \
    --pn_adapt_frequency=10 \
    --gamma=0.99 \
    --mem_size=1000000 \
    --prioritized_replay \
    --alpha=0.3 \
    --beta=1. \
    --no-ranked \
    --no-unreal \
    --wd_scale=1e-3 \
    --n_step_returns \
    --n=60 \
    --add_demos_to_mem \
    --expert_path=$3 \
    --num_demos=$4
