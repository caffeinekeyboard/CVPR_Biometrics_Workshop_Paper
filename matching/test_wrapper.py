import numpy as np
import torch
import torch.nn as nn

from matching.fingernet_wrapper import FingerNetWrapper


class DummyFingerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ensure there is at least one parameter so device detection works
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape
        hp, wp = H // 8, W // 8
        device = x.device

        segmentation = torch.rand(B, 1, hp, wp, device=device)
        minutiae_score = torch.rand(B, 1, hp, wp, device=device)
        minutiae_orientation = torch.rand(B, 45, hp, wp, device=device)

        # create one-hot offset maps with argmax indices in [0,7]
        x_off = torch.zeros(B, 8, hp, wp, device=device)
        y_off = torch.zeros(B, 8, hp, wp, device=device)
        xi = torch.randint(0, 8, (B, hp, wp), device=device)
        yi = torch.randint(0, 8, (B, hp, wp), device=device)
        x_off.scatter_(1, xi.unsqueeze(1), 1.0)
        y_off.scatter_(1, yi.unsqueeze(1), 1.0)

        orientation = torch.rand(B, 90, hp, wp, device=device)
        enhanced_real = torch.rand(B, 1, H, W, device=device)

        return {
            'segmentation': segmentation,
            'minutiae_score': minutiae_score,
            'minutiae_orientation': minutiae_orientation,
            'minutiae_x_offset': x_off,
            'minutiae_y_offset': y_off,
            'orientation': orientation,
            'enhanced_real': enhanced_real,
        }


def test_fingernet_wrapper_forward():
    B = 2
    H = 30  # intentionally not divisible by 8 to exercise padding
    W = 26
    x = torch.rand(B, 1, H, W)

    model = DummyFingerNet()
    wrapper = FingerNetWrapper(model)

    outputs = wrapper.forward(x, minutiae_threshold=0.2)

    expected_keys = {'minutiae', 'enhanced_image', 'segmentation_mask', 'orientation_field'}
    assert set(outputs.keys()) == expected_keys

    minutiae = outputs['minutiae']
    assert isinstance(minutiae, list) and len(minutiae) == B
    for m in minutiae:
        assert m.ndim == 2 and m.shape[1] == 4

    enh = outputs['enhanced_image']
    seg = outputs['segmentation_mask']
    ori = outputs['orientation_field']

    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    Hp = H + pad_h
    Wp = W + pad_w

    assert enh.shape == (B, Hp, Wp)
    assert seg.shape == (B, Hp, Wp)
    assert ori.shape == (B, Hp, Wp)

    assert enh.dtype == torch.uint8
    assert seg.dtype == torch.uint8

    # orientation should be zero where segmentation mask is zero
    mask_bool = seg > 0
    assert torch.all(ori[~mask_bool] == 0)


def test_prepare_input_numpy_2d():
    H, W = 32, 32
    arr = (np.random.rand(H, W) * 255).astype(np.float32)
    model = DummyFingerNet()
    wrapper = FingerNetWrapper(model)
    tensor = wrapper.prepare_input(arr)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 4
