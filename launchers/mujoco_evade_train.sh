#!/usr/bin/env bash
# Example: ./mujoco_evade_train.sh <env_id> <num_learners>
cd ..

mpirun -np $2 python -m algorithms.ddpg.run \
    --cuda \
    --env_id=$1 \
    --seed=0 \
    --checkpoint_dir="data/checkpoints" \
    --enable_visdom \
    --visdom_dir="data/visdom" \
    --log_dir="data/logs" \
    --task="train" \
    --algo="ad2d" \
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
    --no-with_layernorm \
    --reward_scale=1. \
    --quantile_emb_dim=32 \
    --num_tau=8 \
    --num_tau_prime=8 \
    --num_tau_tilde=32 \
    --enable_clipped_double \
    --enable_targ_actor_smoothing \
    --actor_update_delay=2 \
    --actor_lr=1e-3 \
    --critic_lr=1e-3 \
    --clip_norm=40. \
    --noise_type="normal_0.2, ou_0.2" \
    --pn_adapt_frequency=10 \
    --gamma=0.99 \
    --mem_size=1000000 \
    --no-prioritized_replay \
    --alpha=0.3 \
    --beta=1. \
    --no-ranked \
    --no-unreal \
    --wd_scale=1e-3 \
    --n_step_returns \
    --n=60 \
    --no-add_demos_to_mem
