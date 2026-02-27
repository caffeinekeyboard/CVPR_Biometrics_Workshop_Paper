import torch
from model.gumnet import GumNet

# 1) check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2) load
ckpt_path = "model/gumnet_2d_best"
model = GumNet(grid_size=8)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()