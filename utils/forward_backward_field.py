import torch


class BasisMLP(torch.nn.Module):
    def __init__(self, freq_enc=26, hidden_dim=256, num_basis=10, seq_len=10):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.basis_mlp = nn.Sequential(
            nn.Linear(freq_enc, hidden_dim*num_basis),
            nn.Linear(hidden_dim*num_basis, hidden_dim*num_basis),
            nn.Linear(hidden_dim*num_basis, hidden_dim*num_basis))
        
        self.rot_head = nn.Linear(hidden_dim*num_basis, 4)
        self.pos_head = nn.Linear(hidden_dim*num_basis, 3)

        self.seq_len = seq_len


    def forward(self, t):




class PositionalEncoding(nn.Module):

    def __init__(self, d_enc: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)/max_len
        # div_term = torch.exp(torch.arange(0, d_enc, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position)
        pe[:, 0, 1::2] = torch.cos(position)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x

