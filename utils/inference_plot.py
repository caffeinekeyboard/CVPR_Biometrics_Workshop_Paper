from PIL import Image
import matplotlib.pyplot as plt

image_paths = [
    "inference_plots/Arch_Only/Noise_Level_0_8x8_200/1_121.png",
    "inference_plots/Double_Loop_Only//Noise_Level_0_8x8_200/1_220.png",
    "inference_plots/Left_Loop_Only/Noise_Level_0_8x8_200/1_229.png",
    "inference_plots/Natural/Noise_Level_0_8x8_200/1_216.png",
    "inference_plots/Right_Loop_Only/Noise_Level_0_8x8_200/1_891.png",
    "inference_plots/Tented_Arch_Only/Noise_Level_0_8x8_200/1_374.png",
    "inference_plots/Whorl_Only/Noise_Level_0_8x8_200/1_679.png"
]

labels = [
    "Arch",
    "Double Loop",
    "Left Loop",
    "Natural",
    "Right Loop",
    "Tented Arch",
    "Whorl"
]

images = [Image.open(path) for path in image_paths]
width_px, height_px = images[0].size
n = len(images)
dpi = 100
image_width_inches = width_px / dpi
image_height_inches = height_px / dpi
label_width_inches = 1.5 
fig_width = label_width_inches + image_width_inches
fig_height = image_height_inches * n
fig, axes = plt.subplots(nrows=n, ncols=1,
                         figsize=(fig_width, fig_height),
                         dpi=dpi)
if n == 1:
    axes = [axes]
for ax, img, label in zip(axes, images, labels):
    ax.imshow(img)
    ax.axis("off")

    ax.text(
        -0.05, 0.5, label,
        transform=ax.transAxes,
        fontsize=14,  
        va="center",
        ha="right"
    )
left_margin_ratio = label_width_inches / fig_width
plt.subplots_adjust(
    left=left_margin_ratio,
    right=1.0,
    top=1.0,
    bottom=0.0,
    hspace=0.0
)

plt.show()