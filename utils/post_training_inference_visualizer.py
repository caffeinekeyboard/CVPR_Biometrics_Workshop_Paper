import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

from model.gumnet import GumNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/gumnet_2d_best_noise_level_10_4x4.pth'
OUTPUT_DIR = './inference_plots'
TARGET_NOISE_LEVEL = 'Noise_Level_10'
GRID_SIZE = 4  # Adjust this to match your checkpoint

def load_and_preprocess_image(image_path):
    transform = T.Compose([
        T.Grayscale(),
        T.Resize((192, 192)),
        T.ToTensor(),
        T.RandomInvert(p=1.0) 
    ])
    img = Image.open(image_path)
    tensor = transform(img).unsqueeze(0)
    
    return tensor.to(DEVICE)

def plot_results(template, impression, warped_impression):
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    
    axes[0].imshow(template, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Template (Target)")
    axes[0].axis('off')
    
    axes[1].imshow(impression, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Impression (Before)")
    axes[1].axis('off')
    
    axes[2].imshow(warped_impression, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Warped Impression (After)")
    axes[2].axis('off')
    
    overlay_initial = torch.zeros((192, 192, 3))
    overlay_initial[:, :, 0] = torch.tensor(template)
    overlay_initial[:, :, 1] = torch.tensor(impression)
    
    axes[3].imshow(overlay_initial.numpy())
    axes[3].set_title("Initial Overlay\n(Red=Target, Green=Before)")
    axes[3].axis('off')

    overlay_deformation = torch.zeros((192, 192, 3))
    overlay_deformation[:, :, 0] = torch.tensor(impression)
    overlay_deformation[:, :, 1] = torch.tensor(warped_impression)
    
    axes[4].imshow(overlay_deformation.numpy())
    axes[4].set_title("Deformation Overlay\n(Red=Before, Green=After)")
    axes[4].axis('off')

    overlay_final = torch.zeros((192, 192, 3))
    overlay_final[:, :, 0] = torch.tensor(template)
    overlay_final[:, :, 1] = torch.tensor(warped_impression)
    
    axes[5].imshow(overlay_final.numpy())
    axes[5].set_title("Final Alignment\n(Red=Target, Green=After)")
    axes[5].axis('off')
    
    plt.tight_layout()
    return fig

def main(template_path, impression_path, save_path=None):
    
    model = GumNet(in_channels=1, grid_size=GRID_SIZE).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Successfully loaded weights from {CHECKPOINT_PATH}")
    else:
        print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found. Running with untrained weights!")
        
    model.eval()
    Sa = load_and_preprocess_image(template_path)
    Sb = load_and_preprocess_image(impression_path)
    
    with torch.no_grad():
        warped_Sb, _ = model(Sa, Sb)
        
    Sa_plot = Sa.squeeze().cpu().numpy()
    Sb_plot = Sb.squeeze().cpu().numpy()
    warped_Sb_plot = warped_Sb.squeeze().cpu().numpy()
    fig = plot_results(Sa_plot, Sb_plot, warped_Sb_plot)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def process_all_test_images():
    """Process all images in test directories for a specific noise level and save plots."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data_dir = Path('./data')
    
    # Iterate through all categories
    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        print(f"\nProcessing category: {category_name}")
        
        # Check if master directory exists
        master_dir = category_dir / 'master'
        if not master_dir.exists():
            print(f"  No master directory found for {category_name}")
            continue
        
        # Target the specific noise level
        noise_dir = category_dir / TARGET_NOISE_LEVEL
        
        if not noise_dir.exists():
            print(f"  {TARGET_NOISE_LEVEL} not found for {category_name}")
            continue
        
        test_dir = noise_dir / 'test'
        
        if not test_dir.exists():
            print(f"  Test directory not found in {TARGET_NOISE_LEVEL} for {category_name}")
            continue
        
        print(f"  Processing {TARGET_NOISE_LEVEL}...")
        
        # Process each test image
        for test_image_path in sorted(test_dir.glob('*.png')):
            # Extract the base name (e.g., "1_805")
            test_image_name = test_image_path.stem
            # Extract template id (first part before underscore)
            template_id = test_image_name.split('_')[0]
            
            # Look for corresponding template in master directory
            template_path = master_dir / f'{template_id}.png'
            
            if not template_path.exists():
                print(f"    Warning: Template {template_path} not found for {test_image_path.name}")
                continue
            
            # Create output path
            output_path = Path(OUTPUT_DIR) / category_name / (TARGET_NOISE_LEVEL + '_4x4') / f'{test_image_name}.png'
            
            try:
                main(str(template_path), str(test_image_path), str(output_path))
            except Exception as e:
                print(f"    Error processing {test_image_path}: {e}")

if __name__ == '__main__':
    process_all_test_images()