import os
import pytest
import torch
from model import FingerNet

# Prevent thread-pool deadlocks / excessive CPU contention when running under pytest
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@pytest.fixture
def fingernet_model():
    """Fixture to load the FingerNet model."""
    model = FingerNet()
    model.load_state_dict(torch.load('Model.pth', map_location='cpu'))
    model.eval()
    return model


class TestFingerNetModel:
    """Test suite for FingerNet model."""

    def test_model_loading(self):
        """Test that the model loads correctly from checkpoint."""
        model = FingerNet()
        model.load_state_dict(torch.load('Model.pth', map_location='cpu'))
        assert model is not None
        assert isinstance(model, FingerNet)

    def test_model_in_eval_mode(self, fingernet_model):
        """Test that model is set to evaluation mode."""
        assert not fingernet_model.training
        fingernet_model.train()
        assert fingernet_model.training
        fingernet_model.eval()
        assert not fingernet_model.training

    def test_model_parameters_count(self, fingernet_model):
        """Test the total number of model parameters."""
        total_params = sum(p.numel() for p in fingernet_model.parameters())
        trainable_params = sum(p.numel() for p in fingernet_model.parameters() if p.requires_grad)
        
        assert total_params == 4_709_694, f"Expected 4,709,694 parameters, got {total_params}"
        assert trainable_params == total_params, "All parameters should be trainable"

    def test_model_has_required_modules(self, fingernet_model):
        """Test that model contains all required submodules."""
        assert hasattr(fingernet_model, 'img_norm')
        assert hasattr(fingernet_model, 'feature_extractor')
        assert hasattr(fingernet_model, 'ori_seg_head')
        assert hasattr(fingernet_model, 'enhancement_module')
        assert hasattr(fingernet_model, 'minutiae_head')

    @pytest.mark.parametrize("input_shape", [
        (1, 1, 512, 512),
        (2, 1, 512, 512),
        (1, 1, 256, 256),
    ])
    def test_forward_pass_different_batch_sizes(self, fingernet_model, input_shape):
        """Test forward pass with different batch sizes and image sizes."""
        dummy_input = torch.randn(*input_shape)
        
        with torch.no_grad():
            output = fingernet_model(dummy_input)
        
        assert isinstance(output, dict)
        batch_size, _, height, width = input_shape
        scale_factor = height // 64  # Feature map is downsampled by 8x
        
        # Check output shapes
        assert output['orientation upsample'].shape == (batch_size, 90, height, width)
        assert output['segmentation upsample'].shape == (batch_size, 1, height, width)
        assert output['segmentation'].shape == (batch_size, 1, height // 8, width // 8)
        assert output['orientation'].shape == (batch_size, 90, height // 8, width // 8)
        assert output['enhanced_real'].shape == (batch_size, 1, height, width)
        assert output['enhanced_phase'].shape == (batch_size, 1, height, width)

    def test_output_keys(self, fingernet_model):
        """Test that forward pass returns all expected output keys."""
        dummy_input = torch.randn(1, 1, 512, 512)
        
        with torch.no_grad():
            output = fingernet_model(dummy_input)
        
        expected_keys = {
            'orientation upsample',
            'segmentation upsample',
            'segmentation',
            'orientation',
            'enhanced_real',
            'enhanced_phase',
            'minutiae_orientation',
            'minutiae_x_offset',
            'minutiae_y_offset',
            'minutiae_score'
        }
        
        assert set(output.keys()) == expected_keys, f"Unexpected output keys: {set(output.keys())}"

    def test_output_shapes_standard_input(self, fingernet_model):
        """Test output shapes with standard 512x512 input."""
        dummy_input = torch.randn(1, 1, 512, 512)
        
        with torch.no_grad():
            output = fingernet_model(dummy_input)
        
        expected_shapes = {
            'orientation upsample': (1, 90, 512, 512),
            'segmentation upsample': (1, 1, 512, 512),
            'segmentation': (1, 1, 64, 64),
            'orientation': (1, 90, 64, 64),
            'enhanced_real': (1, 1, 512, 512),
            'enhanced_phase': (1, 1, 512, 512),
            'minutiae_orientation': (1, 180, 64, 64),
            'minutiae_x_offset': (1, 8, 64, 64),
            'minutiae_y_offset': (1, 8, 64, 64),
            'minutiae_score': (1, 1, 64, 64),
        }
        
        for key, expected_shape in expected_shapes.items():
            assert output[key].shape == expected_shape, \
                f"Output '{key}' has shape {output[key].shape}, expected {expected_shape}"

    def test_output_values_in_valid_range(self, fingernet_model):
        """Test that output values are in valid ranges (especially for sigmoid outputs)."""
        dummy_input = torch.randn(1, 1, 512, 512)
        
        with torch.no_grad():
            output = fingernet_model(dummy_input)
        
        # Check that sigmoid outputs are in [0, 1]
        sigmoid_outputs = [
            'orientation upsample',
            'segmentation upsample',
            'segmentation',
            'orientation',
            'minutiae_orientation',
            'minutiae_x_offset',
            'minutiae_y_offset',
            'minutiae_score'
        ]
        
        for key in sigmoid_outputs:
            assert torch.all(output[key] >= 0), f"{key} has values < 0"
            assert torch.all(output[key] <= 1), f"{key} has values > 1"

    def test_no_grad_during_inference(self, fingernet_model):
        """Test that gradients are not computed during inference."""
        dummy_input = torch.randn(1, 1, 512, 512, requires_grad=True)
        
        with torch.no_grad():
            output = fingernet_model(dummy_input)
        
        # Check that output tensors don't require grad
        for value in output.values():
            assert not value.requires_grad, "Output should not require gradients"

    def test_segment_method(self, fingernet_model):
        """Test the segment() method returns only segmentation map."""
        dummy_input = torch.randn(1, 1, 512, 512)
        
        with torch.no_grad():
            seg_output = fingernet_model.segment(dummy_input)
        
        assert isinstance(seg_output, torch.Tensor)
        assert seg_output.shape == (1, 1, 512, 512)
        assert torch.all(seg_output >= -1) and torch.all(seg_output <= 1)  # softsign range

    def test_enhance_method(self, fingernet_model):
        """Test the enhance() method returns only enhanced image."""
        dummy_input = torch.randn(1, 1, 512, 512)
        
        with torch.no_grad():
            enh_output = fingernet_model.enhance(dummy_input)
        
        assert isinstance(enh_output, torch.Tensor)
        assert enh_output.shape == (1, 1, 512, 512)