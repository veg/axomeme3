import torch

ckpt_path = "/Users/sergei/Dropbox/TOGA2026/selection_transformer_best.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("Keys in checkpoint:", checkpoint.keys())
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

print("\nSome parameter shapes:")
for k, v in list(state_dict.items())[:15]:
    print(f"  {k}: {v.shape}")
