import torch
import torch.nn as nn


class PersonalizedMotionResidual(nn.Module):
    """
    Identity-specific low-rank motion correction.

    Given the frozen EmoTaG motion feature h:

        z = A h
        delta_exp = s_exp * B_exp z
        delta_jaw = s_jaw * B_jaw z

    Therefore the personalized correction is a strict low-rank
    linear residual whose rank is bounded by `rank`.

    Input:
        encoded_motion: [..., input_dim]

    Output:
        delta_exp: [..., exp_dim]
        delta_jaw: [..., jaw_dim]
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

        if rank <= 0:
            raise ValueError(
                f"LPMR rank must be positive, got {rank}"
            )

        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.exp_dim = int(exp_dim)
        self.jaw_dim = int(jaw_dim)

        # A: R^D -> R^r
        self.down = nn.Linear(
            self.input_dim,
            self.rank,
            bias=False,
        )

        # B_exp: R^r -> R^50
        self.exp_up = nn.Linear(
            self.rank,
            self.exp_dim,
            bias=False,
        )

        # B_jaw: R^r -> R^3
        self.jaw_up = nn.Linear(
            self.rank,
            self.jaw_dim,
            bias=False,
        )

        # Store scales in state_dict so that the main motion checkpoint
        # is self-contained during inference.
        self.register_buffer(
            "exp_scale",
            torch.tensor(
                float(exp_scale),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "jaw_scale",
            torch.tensor(
                float(jaw_scale),
                dtype=torch.float32,
            ),
        )

        # Random initialization for A.
        nn.init.normal_(
            self.down.weight,
            mean=0.0,
            std=0.02,
        )

        # Zero initialization of B makes the complete residual exactly
        # zero at the beginning of personalization.
        nn.init.zeros_(self.exp_up.weight)
        nn.init.zeros_(self.jaw_up.weight)

    def forward(self, encoded_motion):
        # Strict low-rank bottleneck:
        # z = A h
        z = self.down(encoded_motion)

        # delta = B A h
        delta_exp = self.exp_scale * self.exp_up(z)
        delta_jaw = self.jaw_scale * self.jaw_up(z)

        return {
            "delta_exp": delta_exp,
            "delta_jaw": delta_jaw,
        }