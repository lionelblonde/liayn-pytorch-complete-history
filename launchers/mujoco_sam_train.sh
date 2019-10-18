#!/usr/bin/env bash
# Example: ./mujoco_evade_train.sh <env_id> <num_learners> <expert_path> <num_demos>

cd ..

mpiexec -n $2 python main.py \
    --no-cuda \
    --env_id=$1 \
    --seed=$5 \
    --checkpoint_dir="data/checkpoints" \
    --enable_visdom \
    --visdom_dir="data/summaries" \
    --log_dir="data/logs" \
    --task="train" \
    --algo="my" \
    --save_frequency=100 \
    --num_iters=10000000 \
    --training_steps_per_iter=20 \
    --eval_steps_per_iter=50 \
    --eval_frequency=100 \
    --prefill=200 \
    --no-render \
    --rollout_len=5 \
    --batch_size=32 \
    --polyak=0.005 \
    --with_layernorm \
    --reward_scale=1. \
    --quantile_emb_dim=32 \
    --num_tau=4 \
    --num_tau_prime=4 \
    --num_tau_tilde=8 \
    --enable_clipped_double \
    --no-enable_targ_actor_smoothing \
    --actor_update_delay=2 \
    --d_update_ratio=5 \
    --actor_lr=3e-4 \
    --critic_lr=1e-3 \
    --d_lr=3e-4 \
    --clip_norm=5. \
    --minimax_only \
    --no-state_only \
    --noise_type="adaptive-param_0.2, ou_0.2" \
    --pn_adapt_frequency=10 \
    --gamma=0.99 \
    --mem_size=50000 \
    --no-prioritized_replay \
    --alpha=0.3 \
    --beta=1. \
    --no-ranked \
    --no-unreal \
    --wd_scale=1e-4 \
    --n_step_returns \
    --n=60 \
    --ent_reg_scale=0. \
    --no-add_demos_to_mem \
    --expert_path=$3 \
    --num_demos=$4
