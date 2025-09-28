import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406] # keep these for now, in future can recompute with own dataset
IMAGENET_STD  = [0.229, 0.224, 0.225]

# BEFORE RUNNING GET NORM VALS:
WEIGHT_MEAN = 103.97
WEIGHT_STD = 50.09

ACTION_MEAN = torch.tensor([0.0002, 0.0027, 0.0050, -0.0002, 0.0010, -0.0254])
ACTION_STD = torch.tensor([0.0213, 0.0581, 0.0319, 0.0211, 0.0554, 0.0173])

DEPTH_MEAN = [10.0]
DEPTH_STD = [10.0]

class NPZSequenceDataset(Dataset):
    def __init__(self, root_dir, weight_mean, weight_std, action_mean, action_std, depth_man, depth_std):
        self.samples = []

        for subdir in sorted(os.listdir(root_dir)):
            subpath = os.path.join(root_dir, subdir)
            if not os.path.isdir(subpath):
                continue

            info_path = os.path.join(subpath, "dataset_notes")
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

        rotate = ((bin_number - 1) // 3 > 0)
        
        rgb_np = self.resize_with_padding(data_t["rgb"], rotate=rotate)
        rgb_t = torch.from_numpy(rgb_np.astype(np.float32)) / 255.0  # scale to [0,1]
        rgb_t = rgb_t.permute(2, 0, 1)  # convert to (C, H, W)
        rgb_t = TF.normalize(rgb_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        depth_np = self.resize_with_padding(data_t["depth"], rotate=rotate).astype(np.float32)
        depth_t = torch.from_numpy(depth_np).float()
        
        # For depth (likely 2D): add channel dim -> (1, H, W)
        if depth_t.dim() == 2:
            depth_t = depth_t.unsqueeze(0)
        else:
            depth_t = depth_t.permute(2, 0, 1)  # in case depth has channels

        depth_t = TF.normalize(depth_t, DEPTH_MEAN, DEPTH_STD)

        weight_t = torch.from_numpy(data_t["start_weight"]).float().unsqueeze(-1)

        a1_t = data_t["a1"]
        a2_t = data_t["a2"]

        action_t = torch.from_numpy(np.concatenate((a1_t, a2_t), axis=0)).float()
        weight_t1 = torch.from_numpy(data_t1["start_weight"]).float().unsqueeze(-1)

        weight_t = (weight_t - WEIGHT_MEAN) / WEIGHT_STD
        weight_t1 = (weight_t1 - WEIGHT_MEAN) / WEIGHT_STD

        action_t = (action_t - ACTION_MEAN) / ACTION_STD
        return rgb_t, depth_t, weight_t, action_t, weight_t1

