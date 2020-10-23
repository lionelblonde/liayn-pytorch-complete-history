# Official PyTorch implementation of *Lipschitzness Is All You Need*

The repository covers the following algorithms:
* **DDPG, TD3** (Reinforcement Learning)
* **SAM, DAC, DRAM** (Imitation Learning)

Benchmarks and environments covered:
* MuJoCo
* DeepMind Control Suite

## Dependencies

TODO

## Running Experiments
While one can launch any job via `main.py`, it is advised to use `spawner.py`,
designed to spawn a swarm of experiments over multiple seeds and environments in one command.
To get its usage description, type `python spawner.py -h`.
```bash
usage: spawner.py [-h] [--config CONFIG] [--conda_env CONDA_ENV]
                  [--env_bundle ENV_BUNDLE] [--num_workers NUM_WORKERS]
                  [--deployment {tmux,slurm}] [--num_seeds NUM_SEEDS]
                  [--caliber {short,long,verylong,veryverylong}]
                  [--deploy_now] [--no-deploy_now] [--sweep] [--no-sweep]
                  [--wandb_upgrade] [--no-wandb_upgrade]
                  [--num_demos NUM_DEMOS [NUM_DEMOS ...]]

Job Spawner

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG
  --conda_env CONDA_ENV
  --env_bundle ENV_BUNDLE
  --num_workers NUM_WORKERS
  --deployment {tmux,slurm}
                        deploy how?
  --num_seeds NUM_SEEDS
  --caliber {short,long,verylong,veryverylong}
  --deploy_now          deploy immediately?
  --no-deploy_now
  --sweep               hp search?
  --no-sweep
  --wandb_upgrade       upgrade wandb?
  --no-wandb_upgrade
  --num_demos NUM_DEMOS [NUM_DEMOS ...], --list NUM_DEMOS [NUM_DEMOS ...]
```

Here is an example:
```bash
python spawner.py --config tasks/train_mujoco_ddpg.yaml --env_bundle debug --wandb_upgrade --no-sweep --deploy_now --caliber short --num_workers 2 --num_seeds 3 --deployment tmux --conda_env pytorch
```
Check the argument parser in `spawner.py` to know what each of these arguments mean.
