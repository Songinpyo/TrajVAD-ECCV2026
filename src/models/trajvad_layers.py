import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    Causal Convolution 1D.
    Ensures that the output at time t only depends on inputs from time 0 to t.
    Achieved by padding (K-1) * dilation on the left and 0 on the right.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        # x: [B, C, T]
        out = self.conv(x)
        # Remove the extra padding from the right to maintain time dimension T
        return out[:, :, :-self.padding] if self.padding > 0 else out

class TrajVADPoseCoupling(nn.Module):
    """
    Conditional Coupling Layer.
    Predicts Scale and Shift for 'target' based on 'condition'.
    Target is NOT split; it is transformed element-wise based on Condition.
    Here we implement: z_target = (x_target + shift(cond)) * scale(cond)
    
    Args:
        target_dim: Dimension of target tensor.
        cond_dim: Dimension of condition tensor.
        hidden_dim: Hidden layer dimension.
        dilation_pattern: Tuple of (d1, d2) for the two CausalConv layers.
    """
    def __init__(self, target_dim, cond_dim, hidden_dim, dilation_pattern=(1, 2)):
        super().__init__()
        d1, d2 = dilation_pattern
        self.net = nn.Sequential(
            CausalConv1d(cond_dim, hidden_dim, kernel_size=3, dilation=d1),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=d2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, target_dim * 2, kernel_size=1) # Output Shift & Scale per target channel
        )
        
    def forward(self, target, condition):
        # target: [B, target_dim, T]
        # condition: [B, cond_dim, T]
        
        params = self.net(condition)
        shift, scale = params.chunk(2, dim=1)
        
        scale = torch.sigmoid(scale + 2.0) + 1e-6
        
        z = (target + shift) * scale
        logdet = torch.sum(torch.log(scale), dim=[1, 2])
        
        return z, logdet

    def reverse(self, z, condition):
        # z: [B, target_dim, T]
        # condition: [B, cond_dim, T]

        params = self.net(condition)
        shift, scale = params.chunk(2, dim=1)

        scale = torch.sigmoid(scale + 2.0) + 1e-6

        # Inverse affine
        target_old = (z / scale) - shift
        
        # Log determinant (negative for reverse)
        logdet = -torch.sum(torch.log(scale), dim=[1, 2])

        return target_old, logdet

class TrajVADChannelCoupling(nn.Module):
    """
    Channel coupling layer for TrajVAD-T.

    Args:
        left_dim: Left split dimension.
        right_dim: Right split dimension.
        hidden_dim: Hidden layer dimension.
        direction: 'left_to_right' or 'right_to_left'.
        dilation_pattern: Tuple of (d1, d2) for the two CausalConv layers.
    """
    def __init__(self, left_dim, right_dim, hidden_dim, direction='left_to_right', dilation_pattern=(1, 2)):
        super().__init__()
        if direction not in ("left_to_right", "right_to_left"):
            raise ValueError(f"Unsupported direction={direction!r}")
        self.direction = direction
        in_c = left_dim if direction == 'left_to_right' else right_dim
        out_c = right_dim if direction == 'left_to_right' else left_dim
        d1, d2 = dilation_pattern
        
        self.net = nn.Sequential(
            CausalConv1d(in_c, hidden_dim, kernel_size=3, dilation=d1),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=d2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_c * 2, kernel_size=1) # Shift & Scale (1x1 conv)
        )
        
    def forward(self, z_left, z_right):
        if self.direction == 'left_to_right':
            input_feat = z_left
            target_feat = z_right
        else:
            input_feat = z_right
            target_feat = z_left
            
        params = self.net(input_feat)
        shift, scale = params.chunk(2, dim=1)
        
        # Stable sigmoid for scale
        scale = torch.sigmoid(scale + 2.0) + 1e-6
        
        # Affine coupling
        target_new = (target_feat + shift) * scale
        
        # Log determinant
        logdet = torch.sum(torch.log(scale), dim=[1, 2])
        
        if self.direction == 'left_to_right':
            return z_left, target_new, logdet
        else:
            return target_new, z_right, logdet

    def reverse(self, z_left, z_right):
        if self.direction == 'left_to_right':
            input_feat = z_left
            target_feat = z_right
        else:
            input_feat = z_right
            target_feat = z_left
            
        params = self.net(input_feat)
        shift, scale = params.chunk(2, dim=1)
        
        scale = torch.sigmoid(scale + 2.0) + 1e-6
        
        # Inverse affine
        target_old = (target_feat / scale) - shift
        
        # Log determinant (negative for reverse)
        logdet = -torch.sum(torch.log(scale), dim=[1, 2])
        
        if self.direction == 'left_to_right':
            return z_left, target_old, logdet
        else:
            return target_old, z_right, logdet
