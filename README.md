# PyTorch implementations of Distributional RL and IL methods

The repository covers:
* DDPG: [arXiv](https://arxiv.org/abs/1509.02971)
* TD3: [arXiv](https://arxiv.org/abs/1802.09477)
* **TD4-C51**: variant of TD3 with a distributional critic using C51, similarly to D4PG ([openreview](https://openreview.net/pdf?id=SyZipzbCb))
* **D4PG-QR**: idem, but with a Quantile Regression ([arXiv](http://arxiv.org/abs/1710.10044)) critic
* **D4PG-IQN**: idem, but with an IQN ([arXiv](http://arxiv.org/abs/1806.06923)) critic
* SAM: [arXiv](https://arxiv.org/abs/1809.02064)
* **DAM** (variant of SAM built on any variant of TD4)
