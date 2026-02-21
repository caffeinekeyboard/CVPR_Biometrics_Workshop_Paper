import torch.nn as nn
from model.alternate.gumnet_feature_extraction_ap import GumNetFeatureExtractionMP
from model.gumnet_siamese_matching import GumNetSiameseMatching
from model.gumnet_non_linear_alignment import GumNetNonLinearAlignment

class GumNet(nn.Module):
    """
    Complete GumNet Max Pooling architecture for fingerprint matching with non-linear spatial alignment.

    Args:
        in_channels (int, optional): Number of input channels for images. Defaults to 1.

    Shape:
        - Input template: `(B, in_channels, 192, 192)`
        - Input impression: `(B, in_channels, 192, 192)`
        - Output warped_impression: `(B, in_channels, 192, 192)`
        - Output control_points: `(B, 2, grid_size, grid_size)` (default grid_size=4)

    Notes:
        The spatial alignment module returns the warped impression and the predicted
        control-point displacements (not an affine matrix). The default `grid_size`
        is 4, so the control points shape is `(B, 2, 4, 4)` unless configured
        otherwise in `GumNetNonLinearAlignment`.

    Examples:
        >>> model = GumNet()
        >>> template = torch.randn(4, 1, 192, 192)
        >>> impression = torch.randn(4, 1, 192, 192)
        >>> warped, control_points = model(template, impression)
        >>> print(warped.shape, control_points.shape)
        torch.Size([4, 1, 192, 192]) torch.Size([4, 2, 4, 4])
    """

    def __init__(self, in_channels=1, grid_size=4):
        super(GumNet, self).__init__()

        self.feature_extractor = GumNetFeatureExtractionMP(in_channels=in_channels)
        self.siamese_matcher = GumNetSiameseMatching()
        self.spatial_aligner = GumNetNonLinearAlignment(grid_size=grid_size)

    def forward(self, template, impression):
        """
        Forward pass through the complete GumNet.

        Args:
            template (torch.Tensor): Template fingerprint image, shape (B, C, 192, 192)
            impression (torch.Tensor): Impression fingerprint image to be aligned, shape (B, C, 192, 192)

        Returns:
            warped_impression (torch.Tensor): Spatially aligned impression, shape (B, C, 192, 192)
            control_points (torch.Tensor): Predicted control-point displacements, shape (B, 2, grid_size, grid_size).
        """

        # Feature Extraction Module
        template_features = self.feature_extractor(template, branch='Sa')      # [B, 512, 14, 14]
        impression_features = self.feature_extractor(impression, branch='Sb')  # [B, 512, 14, 14]

        # Siamese Matching Module
        corr_ab, corr_ba = self.siamese_matcher(template_features, impression_features)  # [B, 4096] each

        # Spatial Alignment Module
        warped_impression, control_points = self.spatial_aligner(corr_ab, corr_ba, impression) # [B, 1, 192, 192], [B, 2, grid_size, grid_size]

        return warped_impression, control_points
