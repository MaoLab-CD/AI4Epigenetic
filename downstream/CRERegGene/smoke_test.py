import torch

from pretrain.model import mae_bin_base
from downstream.CRERegGene.model import CRERegGeneModel, regression_loss


encoder = mae_bin_base(feature_count=288)
model = CRERegGeneModel(encoder, depth=1, heads=8)
features = torch.rand(2, 3, 288)
role = torch.tensor([[0, 1, 2], [1, 0, 0]])
valid = torch.tensor([[True, True, True], [True, False, False]])
target = torch.rand(2, 4)
prediction = model(features, role, valid)
loss = regression_loss(prediction, target)
loss.backward()
print(f"CRERegGene smoke test passed: shape={tuple(prediction.shape)} loss={loss.item():.6f}")
