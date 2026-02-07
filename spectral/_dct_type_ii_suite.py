import torch
import torch.nn as nn
import torch.fft as fft


def dct2(x, norm="ortho"):
    """
    Compute the Discrete Cosine Transform (DCT) Type II of the input tensor along its last dimension.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., N).
        norm (str, optional): Normalization mode. Default is "ortho".
        
    Returns:
        torch.Tensor: DCT Type II transformed tensor of the same shape as input.
        
    This function computes the DCT Type II by leveraging the Fast Fourier Transform (FFT) using the algorithm described by J. Makhoul in 1980.
    For more information on the theory behind this implementation, refer to: https://doi.org/10.1109/TASSP.1980.1163351
    """
    x_shape = x.shape
    signal_length = x.shape[-1]
    signal_list = x.reshape(-1, signal_length)
    markhoul_shuffled_signal_list = torch.cat([signal_list[:, ::2], signal_list[:, 1::2].flip([1])], dim=1)
    fourier_transformed_signal_list = fft.fft(markhoul_shuffled_signal_list, dim=1)
    twiddle_factor_indices = - torch.arange(signal_length, dtype = x.dtype, device=x.device).unsqueeze(0) * torch.pi / (2 * signal_length)
    twiddle_factor = torch.exp(1j * twiddle_factor_indices)
    final_signal_list = (fourier_transformed_signal_list * twiddle_factor).real
    
    if norm == "ortho":
        final_signal_list[:, 0] /= 2.0 * signal_length ** 0.5
        final_signal_list[:, 1:] /= 2.0 * (signal_length / 2.0)**0.5
        
    output_signals = 2 * final_signal_list.view(*x_shape)
    return output_signals


def idct2(x, norm="ortho"):
    """
    Compute the Inverse Discrete Cosine Transform (IDCT) for DCT-II of the input tensor along its last dimension.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., N).
        norm (str, optional): Normalization mode. Default is "ortho".
    
    Returns:
        torch.Tensor: Inverse DCT Type II transformed tensor of the same shape as input.
        
    It is important to note that the inverse DCT Type II is equivalent to a scaled DCT Type III.
    For more information on the theory behind this implementation, refer to: https://doi.org/10.1109/TASSP.1980.1163351
    """
    x_shape = x.shape
    signal_length = x.shape[-1]
    signal_list = x.reshape(-1, signal_length) / 2.0
    
    if norm == "ortho":
        signal_list[:, 0] *= 2.0 * signal_length ** 0.5
        signal_list[:, 1:] *= 2.0 * (signal_length / 2.0)**0.5
    
    twiddle_factor_indices = torch.arange(signal_length, dtype = x.dtype, device=x.device).unsqueeze(0) * torch.pi / (2 * signal_length)
    twiddle_factor = torch.exp(1j * twiddle_factor_indices)
    signal_list_imag = torch.cat([torch.zeros_like(signal_list[:, :1]), -signal_list[:, 1:].flip([1])], dim=1)
    signal_list_complex = signal_list + 1j * signal_list_imag
    final_signal_list_complex = signal_list_complex * twiddle_factor
    output_signals_shuffled = fft.ifft(final_signal_list_complex, dim=1).real
    output_signal = torch.zeros_like(output_signals_shuffled)
    even_signal_length = (signal_length + 1) // 2
    output_signal[:, ::2] = output_signals_shuffled[:, :even_signal_length]
    output_signal[:, 1::2] = output_signals_shuffled[:, even_signal_length:].flip([1])
    output_signal = output_signal.view(*x_shape)
    return output_signal


def dct2_2d(x, norm=None):
    """
    Compute the Discrete Cosine Transform (DCT) Type II of a two-dimensional input tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., N).
        norm (str, optional): Normalization mode. Default is "ortho".
        
    Returns:
        torch.Tensor: Inverse DCT Type II transformed two-dimensional tensor of the same shape as input.
    """
    X1 = dct2(x, norm=norm)
    X2 = dct2(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)




def idct2_2d(X, norm=None):
    """
    Compute the Inverse Discrete Cosine Transform (IDCT) for DCT-II of a two-dimensional input tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., N).
        norm (str, optional): Normalization mode. Default is "ortho".
        
    Returns:
        torch.Tensor: Inverse DCT Type II transformed two-dimensional tensor of the same shape as input.
    """
    x1 = idct2(X, norm=norm)
    x2 = idct2(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)




class LinearDCT(nn.Linear):
    """
    LinearDCT class implements the 1-D DCT Type-II as a linear layer by calculating the linear transfomration matrix using the DCT Type-II algorithmic suite.
    
    Args:
        in_features (int): Number of input features to the linear layer.
        type (str): 1-D DCT mode among 'dct' and 'idct' for discrete cosine transform and inverse discrete cosine transform respectively.
        
    Returns:
        torch.nn.Linear: A linear layer that takes applies the selected type of transform to the input feature vector.
        
    In practice this layer executes around 50x faster on a GPU since matrix multiplication is one of the most optimized algorithms in parallel processing.
    The drawback is that the DCT matrix will be stored, which increases memory usage.
    """
    def __init__(self, in_features, type, norm='ortho', bias=False):
        self.type = type
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)
        
    def reset_parameters(self):
        """
            This function overrides the reset_parameters() function in nn.Linear class.
            It transforms a 2D identity matrix of dimensions (in_features, in_features) through the selected type of transform.
            This transformed identity matrix is transposed and copied into the weights for this layer and then freezes them.
            
            y = xM where M is the transform matrix, M = dct(I) or idct(I) depending on the selected type of transform.
        """
        I = torch.eye(self.in_features)
        
        if self.type == 'dct':
            transform_matrix = dct2(I, norm=self.norm)
        elif self.type == 'idct':
            transform_matrix = idct2(I, norm=self.norm)
        else:
            raise ValueError("Please select a valid transform type, must be 'dct' or 'idct'.")
        
        with torch.no_grad():
            self.weight.copy_(transform_matrix.t())
            self.weight.requires_grad = False


            

class DCTSpectralPooling(nn.Module):
    """
        DCTSpectralPooling Class implements the spectral pool and filter as a pytorch neural module.
        
        Args:
            height (int): The height of the input feature map.
            width (int): The width of the input feature map.
            keep_h (int): The height of the pre-masked crop in the frequency domain. 
            keep_w (int): The width of the pre-masked crop in the frequency domain.
            
        Returns:
            torch.nn.Module: A pytorch neural module that implements two dimensional DCT Type-II Spectral Pooling on a feature map.
    """
    def __init__(self, height, width, keep_h, keep_w):
        super(DCTSpectralPooling, self).__init__()
        self.dct_h = LinearDCT(height, type='dct', norm='ortho')
        self.dct_w = LinearDCT(width, type='dct', norm='ortho')
        self.idct_h = LinearDCT(height, type='idct', norm='ortho')
        self.idct_w = LinearDCT(width, type='idct', norm='ortho')
        mask = torch.zeros(height, width)
        mask[:keep_h, :keep_w] = 1
        self.register_buffer('mask', mask)
        
    def forward(self, x):
        freq_w = self.dct_w(x)
        freq_w_t = freq_w.transpose(-1, -2)
        freq_hw_t = self.dct_h(freq_w_t)
        freq_2d = freq_hw_t.transpose(-1, -2)
        pooled_freq = freq_2d * self.mask
        pooled_freq_t = pooled_freq.transpose(-1, -2)
        spatial_h_t = self.idct_h(pooled_freq_t)
        spatial_h = spatial_h_t.transpose(-1, -2)
        output = self.idct_w(spatial_h)
        return output