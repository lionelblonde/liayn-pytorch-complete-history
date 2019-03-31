from mpi4py import MPI

import numpy as np
import torch


def mpi_mean_like(x, comm, device):
    """Computes element-wise mean across sbires"""
    num_sbires = comm.Get_size()
    sum_ = torch.empty_like(x).cpu().data.numpy()
    comm.Allreduce(x.cpu().data.numpy(), sum_, op=MPI.SUM)
    return torch.FloatTensor(sum_ / num_sbires).to(device)


def average_gradients(model, comm, device):
    for param in model.parameters():
        mpi_mean_like(param.grad.data, comm, device)


def sync_with_root(model, comm):
    """Send the root node parameters to every sbire"""
    if comm is None:
        return  # do nothing and leave if evaluating
    comm.Barrier()
    rank = comm.Get_rank()
    for param in model.parameters():
        if rank == 0:
            comm.Bcast(param.cpu().data.numpy(), root=0)
        else:
            param_ = np.empty_like(param.cpu().data)
            comm.Bcast(param_, root=0)
            param_ = torch.FloatTensor(param_)
            param.data.copy_(param_.data)
