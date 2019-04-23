"""Implementation of functions and modules for spectral normalization
from the paper https://openreview.net/pdf?id=B1QRgziT-
Note the use of the proposed conputationally-efficient power-iteration method.
The goal is to control the Lipschitz constant of the discriminator function
by constraining the spectral norm of each layer, ie the lagest singular value
of the layer's weight matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(v):
    """Normalize the vector `v`, by default with the l2 norm (for vectors)"""
    return v / (torch.norm(v) + 1e-8)


def normalize_spectrally(weight, u=None, power_iters=2):
    """Power iteration method for weight matrix `weights`"""
    assert power_iters > 0, "power method should iterate at least once"
    # Perform power method iterations
    for _ in range(power_iters):
        v = normalize(torch.mv(weight.t(), u))
        u = normalize(torch.mv(weight, v))
    # Compute the spectral norm as u^T*W*v
    spectral_norm = torch.dot(u, torch.mv(weight, v))
    return spectral_norm, u


class SNLinear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Args:
           in_features: size of each input sample
           out_features: size of each output sample
           bias: If set to False, the layer will not learn an additive bias.
               Default: ``True``
       Shape:
           - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
             additional dimensions
           - Output: :math:`(N, *, out\_features)` where all but the last dimension
             are the same shape as the input.
       Attributes:
           weight: the learnable weights of the module of shape
               `(out_features x in_features)`
           bias: the learnable bias of the module of shape `(out_features)`
           W(Tensor): spectrally normalized weight
           u (Tensor): the right largest singular value of W.

    Note: `SNLinear`'s docstring written in direct analogy with the `Linear`'s docstring.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.zeros(out_features).normal_(0, 1))

    def forward(self, input):
        with torch.no_grad():
            spectral_norm, u = normalize_spectrally(self.weight, self.u)
            self.u.copy_(u)
        return F.linear(input, self.weight / spectral_norm, self.bias)
