from collections import defaultdict
import os.path as osp
import glob
import h5py
import numpy as np
import torch

from helpers import logger


STATS_KEYS = ['ep_lens', 'ep_env_rets']
ALLOW_KEYS = ['obs0', 'acs', 'env_rews', 'dones1', 'obs1'] + STATS_KEYS


def save_dict_h5py(data, fname):
    """Save dictionary containing numpy arrays to h5py file."""
    with h5py.File(fname, 'w') as hf:
        for key in data.keys():
            hf.create_dataset(key, data=data[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    data = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            data[key] = hf[key][()]
    return data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return {k: v[i, ...].astype(np.float32) for k, v in self.data.items()}

    def __len__(self):
        return list(self.data.values())[0].shape[0]


class DemoDataset(Dataset):

    def __init__(self, expert_path, num_demos):
        self.num_demos = num_demos
        self.data = defaultdict(list)
        self.stats = defaultdict(list)
        logger.info(">>>> loading demos")
        # Go over the demos, sorted in alphabetical order
        for i, f in enumerate(sorted(glob.glob(osp.join(expert_path, "*.h5")))):
            # If the desired number of demos has been aggregated, leave
            if i == num_demos:
                break
            # Log the location of the loaded demo
            logger.info("[INFO] demo #{} loaded from: {}".format(str(i).zfill(3), f))
            # Load the demo from the file
            tmp = load_dict_h5py(f)
            dims = {k: tmp[k].shape[1:] for k in tmp.keys()
                    if k not in STATS_KEYS and k in ALLOW_KEYS}
            dims = ' | '.join(["{}={}".format(k, v) for k, v in dims.items()])
            logger.info("[INFO]      dims: {}".format(dims))
            # Collect the demo's content
            for k, v in tmp.items():
                # Add the demo's content
                if k not in ALLOW_KEYS:
                    logger.info("[INFO]      not adding key: {}".format(k))
                    continue
                if k in STATS_KEYS:
                    logger.info("[INFO]      stat: {}{}".format(k.ljust(20, '-'), v))
                    self.stats[k].append(v)
                else:
                    self.data[k].append(v)

        # Transform structures into arrays
        for k, v in self.stats.items():
            self.stats[k] = np.array(v)
        for k, v in self.data.items():
            self.data[k] = np.concatenate(v, axis=0)

        # Log demos' statistics
        rets_, lens_ = (self.stats['ep_env_rets'],
                        self.stats['ep_lens'])
        logger.info("[INFO] got {} transitions, from {} eps".format(len(self), self.num_demos))
        logger.info("[INFO] episodic return: {}({})".format(np.mean(rets_), np.std(rets_)))
        logger.info("[INFO] episodic length: {}({})".format(np.mean(lens_), np.std(lens_)))
