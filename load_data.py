import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPZSequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []  # will hold (f_t, f_t1, bin_number)

        for subdir in sorted(os.listdir(root_dir)):
            subpath = os.path.join(root_dir, subdir)
            if not os.path.isdir(subpath):
                continue

            # read dataset_info.txt if present
            info_path = os.path.join(subpath, "dataset_info.txt")
            bin_number = None
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    for line in f:
                        if line.startswith("Pick bin:"):
                            bin_number = int(line.split(":")[1].strip())
                            break

            # get npz files
            npz_files = sorted([f for f in os.listdir(subpath) if f.endswith(".npz")])
            for i in range(len(npz_files) - 1):
                f_t = os.path.join(subpath, npz_files[i])
                f_t1 = os.path.join(subpath, npz_files[i+1])
                self.samples.append((f_t, f_t1, bin_number))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_t, f_t1, bin_number = self.samples[idx]

        data_t = np.load(f_t)
        data_t1 = np.load(f_t1)

        # do some resizing here
        
        rgb_t = torch.from_numpy(data_t["rgb"]).float()
        depth_t = torch.from_numpy(data_t["depth"]).float()
        weight_t = torch.from_numpy(data_t["weight"]).float()
        action_t = torch.from_numpy(data_t["action"]).float()
        weight_t1 = torch.from_numpy(data_t1["weight"]).float()

        return rgb_t, depth_t, weight_t, action_t, weight_t1, bin_number
