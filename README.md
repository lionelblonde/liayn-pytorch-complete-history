# Deterministic Policy Gradients

PyTorch implementations of notable *deep* extensions of
[Deterministic Policy Gradients](http://proceedings.mlr.press/v32/silver14.pdf).

Covers the following algorithms:
* [**DDPG**: Deep Deterministic Policy Gradients](https://arxiv.org/abs/1509.02971)
* [**TD3**: Twin Delayed DDPG](https://arxiv.org/abs/1802.09477)

Included addons:
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
* [Multi-Step](https://link.springer.com/content/pdf/10.1007%2FBF00114731.pdf)
[returns](https://arxiv.org/abs/1602.01783)
* [UNREAL's Experience Replay](http://arxiv.org/abs/1611.05397)
* [DDPGfD (experts demonstrations in replay buffer)](http://arxiv.org/abs/1707.08817)

# How to

Launching scripts are available in `/launchers`.

## Acknowledgments

Some utilities were inspired from [openai/baselines](https://github.com/openai/baselines),
including the general distribution scheme relying on MPI's *allreduce*
for averaging the gradient updates across parallel optimizers.
The adopted distributed architecture, called *Rapid*, is described in greater details in the
[OpenAI Five blog post](https://openai.com/blog/openai-five/#rapid).
Note that we do not provide support for [NCCL](https://developer.nvidia.com/nccl)'s *allreduce* to
parallelize GPU computations.
Rapid was also used in *Dactyl* to learn [dexterous manipulations](https://openai.com/blog/learning-dexterity/).

![](images/openai_rapid_architecture.png)
