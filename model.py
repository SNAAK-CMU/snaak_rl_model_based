import torch
import torch.nn as nn
import torchvision.models as models


# make sure to resize to 380x380 input
class RGBEnconder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        effecient_net = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(
            effecient_net.features,
            effecient_net.avgpool
            ) # use all but last FC layer
        in_features = effecient_net.classifier[1].in_features # number of features that go into first layer of normal effecient net architecture
        self.fc = nn.Linear(in_features, output_dim)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x)
        x = self.fc(x) # convert to fc of correct size
        return x

# Resize depth to something smaller, efficientnet-B4 has a 380x380 input, so something similar or smaller
class DepthEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))  # makes it input-size agnostic
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        x: (B, 1, H, W)
        """
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.global_pool(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x

class WeightEncoder(nn.Module):
    def __init__(self, output_dim=32):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class ActionEncoder(nn.Module):
    def __init__(self, input_dim = 6, output_dim=32):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class Model(nn.Module):
    def __init__(self, rgb_dim=128, depth_dim=128, weight_dim=16, action_dim=32, hidden_dims=[256, 256, 128, 128]):
        self.rgb_encoder = RGBEnconder(output_dim=rgb_dim)
        self.depth_encoder = DepthEncoder(output_dim=depth_dim)
        self.weight_encoder = WeightEncoder(output_dim=weight_dim)
        self.action_encoder = ActionEncoder(output_dim=action_dim)

        combined_dim = rgb_dim + depth_dim + weight_dim + action_dim
        layers = []
        input_dim = combined_dim
        for output_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.decoder = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)
    
    def forward(self, rgb, depth, weight, action):
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)
        weight_feat = self.weight_encoder(weight)
        action_feat = self.action_encoder(action)

        encoding = torch.concatenate((rgb_feat, depth_feat, weight_feat, action_feat))

        decoding = self.decoder(encoding)
        output = self.output(decoding)
        return output