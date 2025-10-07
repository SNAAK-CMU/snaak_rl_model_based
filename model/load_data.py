import os
import json
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406] # keep these for now, in future can recompute with own dataset
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Normalization stats cache filename within dataset root
STATS_FILENAME = "norm_stats.json"

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
    def __init__(self, root_dir: str, *, recompute_stats: bool = False, stats_path: str | None = None):
        """
        NPZ transitions dataset with on-demand normalization statistics.

        Args:
            root_dir: Path to dataset root which contains dated subfolders of .npz files and a
                'dataset_notes' file specifying the bin number.
            recompute_stats: If True, recompute stats even if a cached JSON file exists.
            stats_path: Optional explicit path to stats JSON; defaults to '<root_dir>/norm_stats.json'.
        """
        self.samples = []
        self.root_dir = root_dir

        # Discover valid (t, t+1) sample pairs with bin numbers
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
                # Require bin info; skip if missing
                if bin_number is None:
                    continue
                try:
                    weight_t = np.load(f_t)["start_weight"]
                    weight_t1 = np.load(f_t1)["start_weight"]
                    # Ensure action keys exist
                    _ = np.load(f_t)["a1"]
                    _ = np.load(f_t)["a2"]
                except Exception:
                    print("Error loading", f_t)
                    # continue

                # Only keep transition if weight decreases or stays the same (with small tolerance)
                if weight_t1 <= weight_t * 1.02 and weight_t - weight_t1 <= 40:
                    if (weight_t1 > weight_t1): # clip to be equal
                        weight_t1 = weight_t
                
                    # if (weight_t - weight_t1) < 5:
                    #     if (np.random.randn() > 0.5): continue # leave out some zeros
                    self.samples.append((f_t, f_t1, bin_number))

        # Load or compute normalization stats once for this dataset
        stats_file = stats_path if stats_path is not None else os.path.join(self.root_dir, STATS_FILENAME)
        self.stats = self._load_or_compute_stats(stats_file, recompute=recompute_stats)
        # Tensors for arithmetic in __getitem__
        self._weight_mean = torch.tensor([self.stats["weight_mean" ]], dtype=torch.float32)
        self._weight_std  = torch.tensor([max(self.stats["weight_std"], 1e-8)], dtype=torch.float32)
        self._action_mean = torch.tensor(self.stats["action_mean"], dtype=torch.float32)
        self._action_std  = torch.tensor([max(s, 1e-8) for s in self.stats["action_std"]], dtype=torch.float32)
        self._rgb_mean = self.stats.get("rgb_mean", IMAGENET_MEAN)
        self._rgb_std  = [max(s, 1e-8) for s in self.stats.get("rgb_std", IMAGENET_STD)]
        self._depth_mean = [float(self.stats.get("depth_mean", 0.0))]
        self._depth_std  = [float(max(self.stats.get("depth_std", 1.0), 1e-8))]
                           
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
        rgb_t = TF.normalize(rgb_t, mean=self._rgb_mean, std=self._rgb_std)

        depth_np = self.crop(data_t["depth"], bin_number, rotate=rotate).astype(np.float32)
        depth_np = np.clip(depth_np, 300, 400)  # clip outliers
        depth_t = torch.from_numpy(depth_np).float()
        
        # For depth (likely 2D): add channel dim -> (1, H, W)
        if depth_t.dim() == 2:
            depth_t = depth_t.unsqueeze(0)
        else:
            depth_t = depth_t.permute(2, 0, 1)  # in case depth has channels

        depth_t = TF.normalize(depth_t, self._depth_mean, self._depth_std)

        weight_t = torch.from_numpy(data_t["start_weight"]).float().unsqueeze(-1)

        a1_t = data_t["a1"]
        a2_t = data_t["a2"]

        action_t = torch.from_numpy(np.concatenate((a1_t, a2_t), axis=0)).float()
        weight_t1 = torch.from_numpy(data_t1["start_weight"]).float().unsqueeze(-1)

        weight_t = (weight_t - self._weight_mean) / self._weight_std
        weight_t1 = (weight_t1 - self._weight_mean) / self._weight_std

        action_t = (action_t - self._action_mean) / self._action_std
        return rgb_t, depth_t, weight_t, action_t, weight_t1

    # -----------------------
    # Stats computation utils
    # -----------------------
    def _load_or_compute_stats(self, stats_file: str, *, recompute: bool = False) -> dict:
        """Load stats from JSON if available; otherwise compute and cache them."""
        if (not recompute) and os.path.exists(stats_file):
            try:
                with open(stats_file, "r") as f:
                    data = json.load(f)
                # Minimal validation
                required = [
                    "rgb_mean", "rgb_std", "depth_mean", "depth_std",
                    "weight_mean", "weight_std", "action_mean", "action_std",
                ]
                if all(k in data for k in required):
                    return data
            except Exception:
                pass  # fall through to recompute

        stats = self._compute_stats()
        try:
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to write stats to {stats_file}: {e}")
        return stats

    def _compute_stats(self) -> dict:
        """Compute dataset normalization statistics by streaming over samples.

        Returns:
            dict with keys: rgb_mean (3), rgb_std (3), depth_mean (float), depth_std (float),
            weight_mean (float), weight_std (float), action_mean (6), action_std (6), meta.
        """
        # Running sums for mean/std: we accumulate sum and sumsq and counts
        def make_acc(dim: int):
            return {
                "count": 0,
                "sum": np.zeros((dim,), dtype=np.float64),
                "sumsq": np.zeros((dim,), dtype=np.float64),
            }

        rgb_acc = make_acc(3)
        depth_acc = make_acc(1)
        weight_acc = make_acc(1)
        action_acc = make_acc(6)

        def update_acc(acc, x_2d: np.ndarray):
            # x_2d shape: (N, D)
            if x_2d.size == 0:
                return
            x64 = x_2d.astype(np.float64)
            acc["count"] += x64.shape[0]
            acc["sum"] += x64.sum(axis=0)
            acc["sumsq"] += (x64 * x64).sum(axis=0)

        used_samples = 0
        for f_t, _f_t1, bin_number in self.samples:
            try:
                data_t = np.load(f_t)
                rotate = ((bin_number - 1) // 3 > 0)

                # RGB in [0,1], per-pixel/channel stats
                rgb_np = self.crop(data_t["rgb"], bin_number, rotate=rotate).astype(np.float32) / 255.0
                h, w, _ = rgb_np.shape
                update_acc(rgb_acc, rgb_np.reshape(h * w, 3))

                # Depth per-pixel stats
                depth_np = self.crop(data_t["depth"], bin_number, rotate=rotate).astype(np.float32)
                depth_np = np.clip(depth_np, 300, 400)
                update_acc(depth_acc, depth_np.reshape(-1, 1))

                # Weight scalar
                w_t = np.float32(data_t["start_weight"]).reshape(1, 1)
                update_acc(weight_acc, w_t)

                # Actions (6-dim)
                a1 = data_t["a1"].reshape(-1)
                a2 = data_t["a2"].reshape(-1)
                act = np.concatenate([a1, a2], axis=0).astype(np.float32).reshape(1, -1)
                if act.shape[-1] != 6:
                    # Fallback: flatten everything to 6 if possible
                    act = act.reshape(1, -1)
                update_acc(action_acc, act)
                used_samples += 1
            except Exception as e:
                print(f"Warning: skipping stats for {f_t}: {e}")
                continue

        def finalize(acc):
            n = max(acc["count"], 1)
            mean = acc["sum"] / n
            var = np.maximum(acc["sumsq"] / n - mean * mean, 0.0)
            std = np.sqrt(var)
            return mean.tolist(), std.tolist()

        rgb_mean, rgb_std = finalize(rgb_acc)
        depth_mean_list, depth_std_list = finalize(depth_acc)
        weight_mean_list, weight_std_list = finalize(weight_acc)
        action_mean, action_std = finalize(action_acc)

        stats = {
            "rgb_mean": rgb_mean,
            "rgb_std": rgb_std,
            "depth_mean": float(depth_mean_list[0]),
            "depth_std": float(depth_std_list[0]),
            "weight_mean": float(weight_mean_list[0]),
            "weight_std": float(weight_std_list[0]),
            "action_mean": action_mean,
            "action_std": action_std,
            "meta": {
                "computed_at": datetime.utcnow().isoformat() + "Z",
                "num_samples_used": used_samples,
                "version": 1,
            },
        }
        return stats

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = NPZSequenceDataset("../rl_dataset", recompute_stats=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    dataset._load_or_compute_stats("test_stats.json", recompute=True)