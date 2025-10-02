import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406] # keep these for now, in future can recompute with own dataset
IMAGENET_STD  = [0.229, 0.224, 0.225]

# BEFORE RUNNING GET NORM VALS:
WEIGHT_MEAN = 104.62
WEIGHT_STD = 61.09

ACTION_MEAN = torch.tensor([-0.0002, 0.0003, 0.0021, 0.0007, 0.0019, -0.0328]) # uniform sampling
ACTION_STD = torch.tensor([0.0193, 0.0540, 0.0317, 0.0193, 0.0539, 0.0147])

DEPTH_MEAN = [337.65]
DEPTH_STD = [66.147]

BIN2_XMIN = 250
BIN2_YMIN = 0
BIN2_XMAX = 470
BIN2_YMAX = 340

# Bin1 coords
BIN1_XMIN = 240
BIN1_YMIN = 0
BIN1_XMAX = 450
BIN1_YMAX = 330

# Bin3 coords 
BIN3_XMIN = 355 
BIN3_YMIN = 0 
BIN3_XMAX = 565  
BIN3_YMAX = 330 

# Bin 4 coords
BIN4_XMIN = 81
BIN4_YMIN = 66
BIN4_XMAX = 440
BIN4_YMAX = 297

# Bin 5 coords
BIN5_XMIN = 67
BIN5_YMIN = 72
BIN5_XMAX = 434
BIN5_YMAX = 305

# Bin 6 coords
BIN6_XMIN = 69
BIN6_YMIN = 94
BIN6_XMAX = 433
BIN6_YMAX = 316

# BIN_COORDS
BIN_COORDS = [
    [BIN1_XMIN, BIN1_YMIN, BIN1_XMAX, BIN1_YMAX],
    [BIN2_XMIN, BIN2_YMIN, BIN2_XMAX, BIN2_YMAX],
    [BIN3_XMIN, BIN3_YMIN, BIN3_XMAX, BIN3_YMAX],
    [BIN4_XMIN, BIN4_YMIN, BIN4_XMAX, BIN4_YMAX],
    [BIN5_XMIN, BIN5_YMIN, BIN5_XMAX, BIN5_YMAX],
    [BIN6_XMIN, BIN6_YMIN, BIN6_XMAX, BIN6_YMAX],
]

class NPZSequenceDataset(Dataset):
    def __init__(self, root_dir):
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
                weight_t = np.load(f_t)["start_weight"]
                weight_t1 = np.load(f_t1)["start_weight"]
                try:
                    np.load(f_t)["a2"]
                except:
                    print("Error loading", f_t)

                # Only keep transition if weight decreases or stays the same
                if weight_t1 <= weight_t * 1.02 and weight_t - weight_t1 <= 50:
                    self.samples.append((f_t, f_t1, bin_number))
                           
    def crop(self, image, bin_number, rotate=False):
        xmin, ymin, xmax, ymax = BIN_COORDS[bin_number-1]
        cropped = image[ymin:ymax, xmin:xmax]
        # print(f"Cropping to bin {bin_number} with coords: ({xmin}, {ymin}), ({xmax}, {ymax}), rotation: {rotate}, shape : {cropped.shape}")
        if rotate:
            #print("Rotating image for bin", bin_number)
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cropped = cv2.resize(cropped, (224, 340))
        #cv2.imshow("Image", cropped)
        #print(cropped.shape)

        #cv2.waitKey(0)
        return cropped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_t, f_t1, bin_number = self.samples[idx]

        data_t = np.load(f_t)
        data_t1 = np.load(f_t1)

        rotate = ((bin_number - 1) // 3 > 0)
        rgb_np = self.crop(data_t["rgb"], bin_number, rotate=rotate)
        rgb_t = torch.from_numpy(rgb_np.astype(np.float32)) / 255.0  # scale to [0,1]
        rgb_t = rgb_t.permute(2, 0, 1)  # convert to (C, H, W)
        rgb_t = TF.normalize(rgb_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        depth_np = self.crop(data_t["depth"], bin_number, rotate=rotate).astype(np.float32)
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

