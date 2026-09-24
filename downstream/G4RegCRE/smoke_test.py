from importlib import import_module

import torch

from pretrain.model import mae_bin_base


module = import_module("downstream.G4RegCRE.model")
G4ToCREModel = module.G4ToCREModel
reconstruction_loss = module.reconstruction_loss

encoder = mae_bin_base(feature_count=288)
model = G4ToCREModel(encoder, target_dim=288, fusion_depth=1)
g4 = torch.rand(2, 4, 288)
cre = torch.rand(2, 288)
relation = torch.nn.functional.one_hot(
    torch.randint(0, 3, (2, 4)), num_classes=3
).float()
distance = torch.randn(2, 4) * 100_000
cis = torch.ones(2, 4)
hic = torch.rand(2, 4)
perturbation = torch.rand(2, 4, 4)
observed = torch.ones(288, dtype=torch.bool)
prediction = model(
    g4, cre, relation, distance, cis, hic, perturbation, observed, observed
)
loss = reconstruction_loss(prediction, torch.rand_like(prediction))
loss.backward()
print(f"G4RegCRE smoke test passed: shape={tuple(prediction.shape)} loss={loss.item():.6f}")
