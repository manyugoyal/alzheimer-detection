import torch #type:ignore
import torch.nn as nn #type: ignore

class DepthwiseSeparableConv3d(nn.Module):
    # Added a stride parameter to allow for spatial reduction during the convolution
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
        
        # 3-Stage Feature Extraction
        # FIX 1: Apply stride=2 to the very first depthwise conv. 
        # This cuts the spatial dimensions in half BEFORE expanding to 32 channels, saving massive VRAM.
        self.block1 = DepthwiseSeparableConv3d(in_channels=1, out_channels=32, stride=2)
        self.block2 = DepthwiseSeparableConv3d(in_channels=32, out_channels=64)
        self.block3 = DepthwiseSeparableConv3d(in_channels=64, out_channels=128)
        
        # FIX 2: Tighter Adaptive Pool. 2x2x2 yields 8 spatial blocks.
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.flatten = nn.Flatten()
        
        # The new feature size: 128 channels * 2 * 2 * 2 = 1024 dimensions.
        # This is a standard, highly compressible embedding size for multimodal fusion.
        self.feature_dim = 128 * 2 * 2 * 2 
        
        # --- THE PRE-TRAINING HEAD ---
        if self.pretrain_mode:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5), # FIX 3: Added Dropout to prevent memorization
                nn.Linear(self.feature_dim, 3) 
            )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.adaptive_pool(x)
        feature_vector = self.flatten(x)
        
        if self.pretrain_mode:
            logits = self.classifier(feature_vector)
            return logits
            
        return feature_vector

if __name__ == "__main__":
    # Test Pre-train Mode
    model = Alzheimer3DCNN(pretrain_mode=True)
    dummy_input = torch.randn(1, 1, 208, 256, 256) 
    
    output = model(dummy_input)
    print(f"Pre-training Output Shape (Classes): {output.shape}") 
    # Expected: [1, 3]
    
    # Test Feature Extractor Mode
    extractor = Alzheimer3DCNN(pretrain_mode=False)
    features = extractor(dummy_input)
    print(f"Extracted Feature Vector Shape: {features.shape}")
    # Expected: [1, 1024]