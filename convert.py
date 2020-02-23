import os
import os.path as osp
import numpy as np
import h5py


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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Ugly, hard-coded conversion to h5.

stems = ['InvertedPendulum-v2',
         'InvertedDoublePendulum-v2',
         'Reacher-v2',
         'Hopper-v3',
         'Walker2d-v3',
         'HalfCheetah-v3',
         'Ant-v3',
         'Humanoid-v3']

for stem in stems:
    fname = stem + '.trajs_16.npz'
    demo_dir = osp.join('/Users/lionelblonde/Downloads/expert-demos.old/more_demos', stem)
    expert_path = osp.join(demo_dir, fname)
    with np.load(expert_path, allow_pickle=True) as data:
        data_map = {}
        for k, v in data.items():
            data_map[k] = v
            print(v[0].shape)

    new_demo_dir = osp.join('/Users/lionelblonde/Downloads/new_demosxxxx', stem)
    new_ext = '.h5'

    for i in range(16):
        _data_map = {}
        for k, v in data_map.items():
            _data_map.update({k: v[i]})
            print(k, v[i].shape)
        new_fname = stem + '_demo{}'.format(str(i).zfill(3))
        fname = new_fname + new_ext
        new_expert_path = osp.join(new_demo_dir, fname)
        print(new_expert_path)
        os.makedirs(new_demo_dir, exist_ok=True)
        save_dict_h5py(_data_map, new_expert_path)

    retrieved_data_map = load_dict_h5py(new_expert_path)

    for k, v in retrieved_data_map.items():
        print(k, v.shape)

print("Bye.")
