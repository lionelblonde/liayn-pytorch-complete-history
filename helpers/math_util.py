import scipy.signal
import torch


def discount(x, gamma):
    """Compute discounted sum along the 0-th dimension of the `x` ndarray
    Return an ndarray `y` with the same shape as x, satisfying:
        y[t] = x[t] + gamma * x[t+1] + gamma^2 * x[t+2] + ... + gamma^k * x[t+k],
            where k = len(x) - t - 1

    Args:
        x (np.ndarray): 2-D array of floats, time x features
        gamma (float): Discount factor
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def huber_quant_reg_loss(td_errors, quantile, kappa=1.0):
    """Huber regression loss (introduced in 1964) following the definition
    in section 2.3 in the IQN paper (https://arxiv.org/abs/1806.06923).
    The loss involves a disjunction of 2 cases:
        case one: |td_errors| <= kappa
        case two: |td_errors| > kappa
    """
    aux = (0.5 * td_errors ** 2 *
           (torch.abs(td_errors) <= kappa).float() +
           kappa *
           (torch.abs(td_errors) - (0.5 * kappa)) *
           (torch.abs(td_errors) > kappa).float())
    return torch.abs(quantile - ((td_errors.le(0.)).float())) * aux / kappa
