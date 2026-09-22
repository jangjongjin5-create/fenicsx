"""No coordinates, condition labels, time, or hidden surface states enter this net."""
import numpy as np
import torch
from torch import nn


class ClosureNet(nn.Module):
    def __init__(self, concentration_ref, width=32, depth=3):
        super().__init__()
        if concentration_ref <= 0:
            raise ValueError("A fixed, positive physical concentration reference is required")
        self.width, self.depth = width, depth
        self.register_buffer("concentration_ref", torch.tensor(float(concentration_ref), dtype=torch.float64))
        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(3 if i == 0 else width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers).double()
        nn.init.normal_(self.layers[-1].weight, std=0.025)
        nn.init.constant_(self.layers[-1].bias, 2.0)

    def forward(self, normalized_state):
        return torch.sigmoid(self.layers(normalized_state)).squeeze(-1)

    def physical(self, cp, cb, theta):
        """Reusable frozen callable, with physical concentrations in mol/m³."""
        xi = np.stack([cp/float(self.concentration_ref), cb/float(self.concentration_ref), theta], axis=-1)
        with torch.no_grad():
            return self(torch.as_tensor(xi, dtype=torch.float64)).numpy()

    def specification(self):
        return dict(inputs=["cp/c_ref", "cb/c_ref", "theta_p"], width=self.width,
                    depth=self.depth, activation="tanh", output="sigmoid",
                    concentration_ref_mol_m3=float(self.concentration_ref), dtype="float64")


def load_frozen_checkpoint(path):
    """Load a trusted Stage 8 checkpoint for geometry-independent inference."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    spec = checkpoint["specification"]
    net = ClosureNet(spec["concentration_ref_mol_m3"],spec["width"],spec["depth"])
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net
