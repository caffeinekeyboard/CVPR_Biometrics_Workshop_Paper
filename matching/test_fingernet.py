import os
import torch
import torch.nn.functional as F
import cv2
from fingernet import FingerNet
from fingernet_wrapper import FingerNetWrapper
from torchsummary import summary

# Limit threads to avoid hangs during summary/model introspection
# os.environ.setdefault('OMP_NUM_THREADS', '1')
# os.environ.setdefault('MKL_NUM_THREADS', '1')
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

# Resolve paths relative to this script so the script works from any CWD
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_IMAGE_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'assets', 'anguli_fingerprint.png'))
_MODEL_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, 'fingernet.pth'))

im = cv2.imread(_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if im is None:
    raise FileNotFoundError(f"Could not load image at {_IMAGE_PATH}. verify the file exists")

# Convert image to tensor and add batch dimension
im_tensor = torch.from_numpy(im).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

def _pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

def test1():
    model = FingerNet()
    model.load_state_dict(torch.load(_MODEL_PATH, map_location='cpu'))
    model.eval()
    model.to('cpu')
    with torch.no_grad():
        padded = _pad_to_multiple(im_tensor, 8)
        output = model(padded)
    print("FingerNet ran successfully")

def test2():
    model = FingerNet()
    model.load_state_dict(torch.load(_MODEL_PATH, map_location='cpu'))
    model.eval()
    model.to('cpu')

    wrapper = FingerNetWrapper(model)
    wrapper.to('cpu')

    out = wrapper(im_tensor)

    wrapper.plot_minutiae(im_tensor, out, save_path='minutiae_detection.png')


if __name__ == '__main__':
    test2()
    test1()