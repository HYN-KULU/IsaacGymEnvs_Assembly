import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
def make_depth_resnet18(pretrained=False):
    model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
    # Change input to 1-channel
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

# Example usage
depth = np.load("data/depth_0.npy")
depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()  # (1,1,480,640)
depth = (depth - depth.min()) / (depth.max() - depth.min())
print(depth.shape)

encoder = make_depth_resnet18(pretrained=False)
feat = encoder(depth)   # forward pass
print("Feature shape:", feat.shape)
