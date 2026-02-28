import torch
from torchinfo import summary
from model.gumnet import GumNet

model = GumNet(grid_size=8)
input_shapes = [(100, 1, 192, 192), (100, 1, 192, 192)]
print("\n" + "="*50)
print("GUMNET 2D ARCHITECTURE SUMMARY")
print("="*50)

summary(
    model, 
    input_size=input_shapes,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
)