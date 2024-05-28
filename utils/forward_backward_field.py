import torch
import numpy as np
import torch.nn.functional as F


class MotionMLP(torch.nn.Module):
    """Motion trajectory MLP module."""

    def __init__(
        self,
        num_basis=6,
        D=8,
        W=256,
        input_ch=4,
        num_freqs=16,
        skips=[4],
        sf_mag_div=1.0,
        num_output_channels=3
    ):
        """Init function for motion MLP.

        Args:
        num_basis: number motion basis
        D: MLP layers
        W: feature dimention of MLP layers
        input_ch: input number of channels
        num_freqs: number of rquency for position encoding
        skips: where to inject skip connection
        sf_mag_div: motion scaling factor
        """
        super(MotionMLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = int(input_ch + input_ch * num_freqs * 2)
        self.skips = skips
        self.sf_mag_div = sf_mag_div

        self.xyzt_embed = PeriodicEmbed(max_freq=num_freqs, N_freq=num_freqs)

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(self.input_ch, W)]
            + [
                torch.nn.Linear(W, W)
                if i not in self.skips
                else torch.nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)
            ]
        )

        self.coeff_linear = torch.nn.Linear(W, num_basis * num_output_channels)
        self.coeff_linear.weight.data.fill_(0.0)
        self.coeff_linear.bias.data.fill_(0.0)

    def forward(self, x):
        input_pts = self.xyzt_embed(x)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # sf = torch.nn.functional.tanh(self.sf_linear(h))
        pred_coeff = self.coeff_linear(h)

        return pred_coeff / self.sf_mag_div


class PeriodicEmbed(torch.nn.Module):
    """Fourier Position encoding module."""

    def __init__(self, max_freq, N_freq, linspace=True):
        """Init function for position encoding.

        Args:
            max_freq: max frequency band
            N_freq: number of frequency
            linspace: linearly spacing or not
        """
        super().__init__()
        self.embed_functions = [torch.cos, torch.sin]
        if linspace:
            self.freqs = torch.linspace(1, max_freq + 1, steps=N_freq)
        else:
            exps = torch.linspace(0, N_freq - 1, steps=N_freq)
            self.freqs = 2**exps

    def forward(self, x):
        output = [x]
        for f in self.embed_functions:
            for freq in self.freqs:
                output.append(f(freq * x))

        return torch.cat(output, -1)


def init_dct_basis(num_basis, num_frames):
    """Initialize motion basis with DCT coefficient."""
    T = num_frames
    K = num_basis
    dct_basis = torch.zeros([T, K])

    for t in range(T):
        for k in range(1, K + 1):
            dct_basis[t, k - 1] = np.sqrt(2.0 / T) * np.cos(
                np.pi / (2.0 * T) * (2 * t + 1) * k
            )

    return dct_basis


def compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, trajectory_basis_i):
    return torch.cat(
        [
            torch.sum(raw_coeff_x * trajectory_basis_i, axis=-1, keepdim=True),
            torch.sum(raw_coeff_y * trajectory_basis_i, axis=-1, keepdim=True),
            torch.sum(raw_coeff_z * trajectory_basis_i, axis=-1, keepdim=True),
        ],
        dim=-1,
    )


class MotionPredictor(torch.nn.Module):
    def __init__(self, num_basis=6, num_frames=148, device="cuda:0"):
        super(MotionPredictor, self).__init__()
        self.num_frames = num_frames
        self.device = device
        self.motion_mlp = MotionMLP(num_basis=6, num_output_channels=3).to(self.device)
        dct_basis = init_dct_basis(num_basis, num_frames)
        self.trajectory_basis = (
            torch.nn.parameter.Parameter(dct_basis).float().to(self.device).detach().requires_grad_(True)
        )

    def forward(self, means3D, time_idx):
        time = (torch.ones(means3D.shape[0]) * (time_idx/self.num_frames)).to(self.device)
        
        ref_xyzt = (
            torch.cat([means3D, time.unsqueeze(1)], dim=-1)
            .float()
            .to(means3D.device)
        )
        raw_coeff_xyz = self.motion_mlp(ref_xyzt)

        num_basis = self.trajectory_basis.shape[1]
        raw_coeff_x = raw_coeff_xyz[..., 0:num_basis]
        raw_coeff_y = raw_coeff_xyz[..., num_basis : num_basis * 2]
        raw_coeff_z = raw_coeff_xyz[..., num_basis * 2 : num_basis * 3]

        delta_means3D = compute_traj_pts(
            raw_coeff_x,
            raw_coeff_y,
            raw_coeff_z,
            self.trajectory_basis[None, None, time_idx + 1, :],
            )

        return delta_means3D
        
