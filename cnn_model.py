import torch
import torch.nn as nn

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1): 
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.pointwise(self.depthwise(x)))))

class Alzheimer3DCNN(nn.Module):
    def __init__(self, pretrain_mode=True):
        super().__init__()
        self.pretrain_mode = pretrain_mode
        
        # 4-Block architecture matching the training weights
        self.block1 = DepthwiseSeparableConv3d(in_channels=1, out_channels=32, stride=2)
        self.block2 = DepthwiseSeparableConv3d(in_channels=32, out_channels=64)
        self.block3 = DepthwiseSeparableConv3d(in_channels=64, out_channels=128)
        self.block4 = DepthwiseSeparableConv3d(in_channels=128, out_channels=256)
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        
        if self.pretrain_mode:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
                nn.Linear(128, 3) 
            )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        feature_vector = self.flatten(x)
        
        if self.pretrain_mode:
            return self.classifier(feature_vector)
        return feature_vector