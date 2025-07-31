import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # 구현하세요!
        # update gate
        self.W_z = nn.Linear(input_size, hidden_size, bias=True)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        # reset gate
        self.W_r = nn.Linear(input_size, hidden_size, bias=True)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        # candidate
        self.W_h = nn.Linear(input_size, hidden_size, bias=True)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h))
        return (1 - z) * h_tilde + z * h


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        batch_size, seq_len, _ = inputs.size()
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        outputs = []
        for t in range(seq_len):
            h = self.cell(inputs[:, t, :], h)
            outputs.append(h.unsqueeze(1))
        # (batch, seq_len, hidden_size)
        return torch.cat(outputs, dim=1)