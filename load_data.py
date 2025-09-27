import os
from more_itertools import padded
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class NPZSequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for subdir in sorted(os.listdir(root_dir)):
            subpath = os.path.join(root_dir, subdir)
            if not os.path.isdir(subpath):
                continue

            info_path = os.path.join(subpath, "dataset_info.txt")
            bin_number = None
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    for line in f:
                        if line.startswith("Pick Bin:"):
                            bin_number = int(line.split(":")[1].strip())
                            break

            npz_files = sorted([f for f in os.listdir(subpath) if f.endswith(".npz")])
            for i in range(len(npz_files) - 1):
                f_t = os.path.join(subpath, npz_files[i])
                f_t1 = os.path.join(subpath, npz_files[i+1])
                self.samples.append((f_t, f_t1, bin_number))
                           
    def resize_with_padding(self, image, target_size=(380, 380), rotate=False):
        h, w = image.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT)
        if rotate:
            padded = cv2.rotate(padded, cv2.ROTATE_90_CLOCKWISE)
        return padded

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_t, f_t1, bin_number = self.samples[idx]

        data_t = np.load(f_t)
        data_t1 = np.load(f_t1)

        rotate = (data_t["bin_number"] is not None) and ((data_t["bin_number"] - 1) // 3 > 0)
        
        rgb_t = self.resize_with_padding(torch.from_numpy(data_t["rgb"]).float(), rotate)
        depth_t = self.resize_with_padding(torch.from_numpy(data_t["depth"]).float(), rotate)
        weight_t = torch.from_numpy(data_t["weight"]).float()
        action_t = torch.from_numpy(data_t["action"]).float()
        weight_t1 = torch.from_numpy(data_t1["weight"]).float()

        return rgb_t, depth_t, weight_t, action_t, weight_t1, bin_number
