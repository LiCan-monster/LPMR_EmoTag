
import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonalizedMotionResidual(nn.Module):
    """
    Identity-specific low-rank motion correction.

    Input:
        encoded_motion: [B, 256]

    Output:
        delta_exp: [B, 50]
        delta_jaw: [B, 3]

    Only trained during few-shot personalization.
    """

    def __init__(
        self,
        input_dim=256,
        rank=8,
        exp_dim=50,
        jaw_dim=3,
        exp_scale=0.1,
        jaw_scale=0.05,
    ):
        super().__init__()

        self.down = nn.Linear(input_dim, rank, bias=False)

        self.exp_up = nn.Linear(
            rank, exp_dim, bias=False
        )

        self.jaw_up = nn.Linear(
            rank, jaw_dim, bias=False
        )

        self.exp_scale = exp_scale
        self.jaw_scale = jaw_scale

        # Small random down-projection
        nn.init.normal_(
            self.down.weight,
            mean=0.0,
            std=0.02
        )

        # Zero initialization guarantees
        # delta_motion = 0 at iteration 0
        nn.init.zeros_(self.exp_up.weight)
        nn.init.zeros_(self.jaw_up.weight)

    def forward(self, encoded_motion):

        z = self.down(encoded_motion)

        z = F.gelu(z)

        delta_exp = (
            self.exp_scale
            * self.exp_up(z)
        )

        delta_jaw = (
            self.jaw_scale
            * self.jaw_up(z)
        )

        return {
            "delta_exp": delta_exp,
            "delta_jaw": delta_jaw,
        }