import torch
import numpy as np
import time

class RNNCell(torch.nn.Module):
    """A single step RNN in pytorch

    h_t = tanh(x_t * W_x + h_{t-1} * W_h + b)
    """
    def __init__(self, input_size, hidden_size):
        """for a single step

        Args:
            input_size (float): [batch_size, input_size]
            hidden_size (float): [batch_size, hidden_size]
        """
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_x = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_h = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        """forward pass from input to hidden state

        Args:
            x_t (float): [batch_size, input_size]
            h_prev (float): [batch_size, hidden_size]
        """
        h_t = torch.tanh(x_t @ self.W_x + h_prev @ self.W_h + self.b_h)
        return h_t
    
class GRUCell(torch.nn.Module):
    """A single pass through a GRU cell

    z_t = tanh(x_t * W_z + h_{t-1} * U_z + b_z)
    r_t = tanh(x_t * W_r + h_{t-1} * U_r + b_r)
    h~_t = tanh(x_t * W_h + (r_t dot h_{t-1}) * U_h + b_h)
    h_t = (1 - z_t) dot h_{t-1} + z_t dot h~_t
    """
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_z = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_r = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_h = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_z = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_r = torch.nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.U_h = torch.nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_z = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_r = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))
        

    def forward(self, x_t, h_prev):
        """forward pass through GRU

        Args:
            x_t (float): [batch_size, input_size]
            h_prev (float): [batch_size, hidden_size]
        """
        z_t = torch.sigmoid(x_t @ self.W_z + h_prev @ self.U_z + self.b_z)
        r_t = torch.sigmoid(x_t @ self.W_r + h_prev @ self.U_r + self.b_r)
        h_tilda_t = torch.tanh(x_t @ self.W_h + (r_t * h_prev) @ self.U_h + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilda_t
        return h_t


class RNN(torch.nn.Module):
    """RNN network
    """
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = RNNCell(input_size, hidden_size)
        self.gru_cell = GRUCell(input_size, hidden_size)
        
        # output dimension of W_out should be equal to input size for copy task
        self.W_out = torch.nn.Parameter(torch.randn(hidden_size, input_size))
        self.b_out = torch.nn.Parameter(torch.zeros(input_size))

    def forward(self, X, cell_choice='rnn'):
        """forward pass through the RNN

        Args:
            X (float): [batch_size, seq_len, input_size]
            cell_choice (str): rnn/gru
        """
        batch_size, seq_len, _ = X.shape
        # initial hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=X.device)
        outputs = []
        for t in range(seq_len):
            x_t = X[:, t, :]    # [batch_size, input_size]
            h = self.rnn_cell(x_t, h)   # [batch_size, hidden_size]
            # h = self.gru_cell(x_t, h)
            # Project hidden -> input_size
            out_t = h @ self.W_out + self.b_out
            outputs.append(out_t.unsqueeze(1))  # shape [batch_size, 1, input_size]
        # concatenate across time
        return torch.cat(outputs, dim=1)    # [batch_size, seq_length, input_size]
    
if __name__ == '__main__':
    seq_length = 20
    batch_size = 32
    input_size = 10
    hidden_size = 128
    n_epochs = 10
    lr = 0.01
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    np.random.seed(42)
    X_train = np.random.rand(1000, seq_length, input_size).astype(np.float32)
    Y_train = X_train.copy()

    model = RNN(input_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    X_torch = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_torch = torch.tensor(Y_train, dtype=torch.float32, device=device)

    start_time = time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_torch, cell_choice='gru') # [batch_size, seq_length, input_size]
        loss = criterion(output, Y_torch)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} | Loss torch: {loss.item():.6f}")
    
    print(f"computational time: {time.time()-start_time}")