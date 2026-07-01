import torch

weights = torch.load(
    "adni_pilot_weights.pth",
    map_location="cpu",
    weights_only=True
)

for name, param in weights.items():
    print(f"\n{name}")
    print(param)