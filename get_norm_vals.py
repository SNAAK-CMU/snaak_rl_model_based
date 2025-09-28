from torch.utils.data import DataLoader
from load_data import NPZSequenceDataset
import torch

dataset = NPZSequenceDataset("rl_dataset")
loader = DataLoader(dataset, batch_size=64, shuffle=False)

weights = []
actions = []
depths = []

for rgb, depth, weight, action, next_weight in loader:
    weights.append(weight)
    weights.append(next_weight)
    actions.append(action)
    depths.append(depth)

all_weights = torch.cat(weights, dim=0)
all_actions = torch.cat(actions, dim=0)

all_depths = torch.cat(depths, dim=0).view(-1)

weight_mean = all_weights.mean()
weight_std = all_weights.std()

action_mean = all_actions.mean(dim=0)
action_std = all_actions.std(dim=0)

depth_mean = all_depths.mean()
depth_std = all_depths.std()

print("Weight mean/std:", weight_mean.item(), weight_std.item())
print("Action mean/std:", action_mean, action_std)
print("Depth mean/std:", depth_mean.item(), depth_std.item())
