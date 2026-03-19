from torch import nn
import torch
import numpy as np

class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        B = torch.randn((num_input_channels, mapping_size)) * scale
        B = B[B.norm(dim=1).argsort()]
        self.register_buffer('_B', B)

    def forward(self, x):
        batches, channels = x.shape

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        res = x @ self._B

        res = res.mul_(2 * torch.pi)
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)

class FeatureMLP(nn.Module):
    def __init__(self, depth, width, out_dim, input_dim=3, positional_encoding=False, sigma=5.0,
                 normalize=True):
        super(FeatureMLP, self).__init__()

        self.normalize = normalize
        layers = []
        if positional_encoding:
            layers.append(FourierFeatureTransform(input_dim, width, sigma))
            layers.append(nn.Linear(width * 2 + input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        else:
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        layers.append(nn.Linear(width, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)

        if self.normalize:
            # Unit norm output
            x = torch.nn.functional.normalize(x, dim=-1)

        return x