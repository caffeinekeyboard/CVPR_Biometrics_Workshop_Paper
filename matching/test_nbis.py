import nbis
from nbis import NbisExtractorSettings
import os
import torchvision.transforms as transforms
from nbis_extractor import Nbis
import torch
import numpy as np
import cv2


# Resolve paths relative to this script so the script works from any CWD
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_IMAGE_PATH = os.path.abspath(
    os.path.join(
        _SCRIPT_DIR,
        '..',
        'data',
        'FCV',
        'FVC2004',
        'Dbs',
        'DB1_A',
        '1_1.tif',
    )
)


_TRANSFORM = transforms.Compose([
    #transforms.Grayscale(num_output_channels=1),
    transforms.Pad(padding=(62, 0, 63, 0), fill=255),
    transforms.Resize((192, 192)),
    #transforms.RandomInvert(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def _load_single_image_tensor() -> "torch.Tensor":

    from PIL import Image

    im = Image.open(_IMAGE_PATH).convert("L")
    im_tensor = _TRANSFORM(im).unsqueeze(0) # type: ignore
    return im_tensor


def _save_minutiae_visualization(
    original_image: torch.Tensor,
    outputs: dict,
    save_path: str,
    denormalize: bool = True,
    mean: float = 0.5,
    std: float = 0.5,
):
    """
    Save the original image with detected minutiae points (NBIS).
    """
    if original_image.ndim == 4:
        img = original_image[0, 0].cpu().numpy()
    elif original_image.ndim == 3:
        img = original_image[0].cpu().numpy()
    elif original_image.ndim == 2:
        img = original_image.cpu().numpy()
    else:
        raise ValueError(f"Unexpected image shape: {original_image.shape}")

    img = img.astype(np.float32)
    if denormalize and img.max() <= 1.5:
        if img.min() < 0:
            img = img * std + mean
        img = np.clip(img, 0.0, 1.0)
        img_norm = (img * 255.0).astype(np.uint8)
    else:
        img_norm = np.clip(img, 0.0, 255.0).astype(np.uint8)

    minutiae = outputs['minutiae'][0].cpu().numpy()

    h, w = img_norm.shape
    canvas = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

    shown_count = 0
    if len(minutiae) > 0:
        for i in range(len(minutiae)):
            x = int(minutiae[i, 0])
            y = int(minutiae[i, 1])
            angle = float(minutiae[i, 2])

            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            shown_count += 1

            cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)

            arrow_length = 15
            end_x = int(x + arrow_length * np.cos(angle))
            end_y = int(y + arrow_length * np.sin(angle))
            cv2.arrowedLine(canvas, (x, y), (end_x, end_y), (0, 255, 255), 1, tipLength=0.3)

    label = f"Minutiae: {shown_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size
    margin = 4
    x0, y0 = 5, 5
    x1, y1 = x0 + text_w + 2 * margin, y0 + text_h + 2 * margin
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(canvas, label, (x0 + margin, y0 + text_h + margin),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imwrite(save_path, canvas)
    print(f"Minutiae visualization saved to {save_path}")


def test_single_image():

    im_tensor = _load_single_image_tensor()
    extractor = Nbis()

    result = extractor(im_tensor)

    assert result is not None
    save_path = os.path.join(_SCRIPT_DIR, 'nbis_minutiae.png')
    _save_minutiae_visualization(im_tensor, result, save_path)
    print("NBIS single-image extraction ran successfully")


def test_batch_images(batch_size: int = 8):

    im_tensor = _load_single_image_tensor().repeat(batch_size, 1, 1, 1)
    extractor = Nbis()
    
    results = extractor(im_tensor)

    assert (results is not None)
    print("NBIS batch extraction ran successfully")


if __name__ == '__main__':
    test_single_image()
    test_batch_images()