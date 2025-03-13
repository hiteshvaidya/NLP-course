import torch
import numpy as np
import time
import pickle as pkl

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

class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_f = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_i = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_c = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_o = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_f = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_i = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_c = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_o = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev, c_prev):
        f_t = torch.sigmoid(x_t @ self.W_f + h_prev @ self.U_f + self.b_f)
        i_t = torch.sigmoid(x_t @ self.W_i + h_prev @ self.U_i + self.b_i)
        c_tilda_t = torch.tanh(x_t @ self.W_c + h_prev @ self.U_c + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilda_t
        o_t = torch.sigmoid(x_t @ self.W_o + h_prev @ self.U_o + self.b_o)
        h_t = o_t * torch.tanh(c_t)
        return h_t
    
class mLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_i = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_m = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_i = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_m = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_f = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_f = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_o = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_o = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_c = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_c = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.b_m = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev, c_prev):
        m_t = x_t @ self.W_mx + h_prev @ self.U_mh + self.b_m
        M_t = torch.diag(m_t)
        x_tilde_t = x_t @ M_t
        i_t = torch.sigmoid(x_tilde_t @ self.W_i + h_prev @ self.U_i + self.b_i)
        f_t = torch.sigmoid(x_t @ self.W_f + h_prev @ self.U_f + self.b_f)
        o_t = torch.sigmoid(x_t @ self.W_o + h_prev @ self.U_o + self.b_o)
        c_tilde_t = torch.tanh(x_t @ self.W_c + h_prev @ self.U_c + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde_t
        h_t = o_t * torch.tanh(c_t)
        return h_t
    
class mGRU(torch.nn.Module):
    """
    source: https://arxiv.org/pdf/1907.00455
    """
    def __init__(self, input_size, hidden_size):
        super(mGRU, self).__init__()
        self.hidden_size = hidden_size
        self.W_m = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_z = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_r = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_h = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_z = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_r = torch.nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.U_h = torch.nn.Parameter(torch.zeros(hidden_size, hidden_size))
        
        self.b_m = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_z = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_r = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        m_t = x_t @ self.W_m + h_prev @ self.U_m + self.b_m
        M_t = torch.diag(m_t)
        x_tilde_t = x_t @ M_t
        z_t = torch.sigmoid(x_tilde_t @ self.W_z + h_prev @ self.U_z + self.b_z)
        r_t = torch.sigmoid(x_tilde_t @ self.W_r + h_prev @ self.U_r + self.b_r)
        h_tilda_t = torch.tanh(x_tilde_t @ self.W_h + (r_t * h_prev) @ self.U_h + self.b_h)
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
    
def generate_sequences(seq_length, padding, vocabulary, 
                       delimiter, unknown, output_len):
    input = []
    output = []
    for index in range(seq_length):
        input.append(np.random.choice(vocabulary))
    output = input.copy()
    for index in range(padding):
        input.append(delimiter)
    for index in range(output_len):
        input.append(unknown)
    output = len(input) * [unknown] + output[:output_len]
    return input, output

    
if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    seq_length = [100, 200, 500, 1000]
    vocabulary = ['a','x','c','r','y','w','b','t','o']
    delimiter = '$'
    unknown = ' '
    vocabulary.extend([delimiter, unknown])
    print(f"vocabulary: {vocabulary}")
    
    char2idx = {char: idx for idx, char in enumerate(vocabulary)}
    idx2char = {idx: char for idx, char in enumerate(vocabulary)}
    idx_vocab = [char2idx[char] for char in vocabulary]
    print(f"idx_vocab: {idx_vocab}")
    idx_vocab.remove(char2idx[delimiter])
    idx_vocab.remove(char2idx[unknown])

    padding = [10, 20, 50]  # repeat delimiter for how many time steps
    output_len = [50, 100, 200] # how many time steps to predict
    batch_size = 4
    input_size = len(vocabulary)
    hidden_size = 128
    n_epochs = 10
    lr = 0.01
    n_samples = 100000

    np.random.seed(42)
    # X_train = np.random.rand(n_samples, seq_length, input_size).astype(np.float32)
    # Y_train = X_train.copy()

    X_train = []
    Y_train = []
    for index in range(n_samples):
        input, output = generate_sequences(seq_length[0], padding=padding[0], vocabulary=idx_vocab, delimiter=char2idx[delimiter], unknown=char2idx[unknown], output_len=output_len[0])
        X_train.append(input)
        Y_train.append(output)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    pkl.dump((X_train, Y_train), open('copyTask_data.pkl', 'wb'))
    print(f"input: {X_train.shape}, output: {Y_train.shape}")
    print(f"input sample: {input[:5]}")
    print(f"output sample: {output[:5]}")

    # (X_train, Y_train) = pkl.load(open('copyTask_data.pkl', 'rb'))

    model = RNN(input_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    embedding_layer = torch.nn.Embedding(num_embeddings=len(idx_vocab)+2, 
                                        embedding_dim=input_size).to(device)

    start_time = time.time()
    for epoch in range(n_epochs):
        for index in range(0, n_samples, batch_size):
            batchX = X_train[index:index+batch_size]
            print(f"batchX: {batchX}")
            batchY = Y_train[index:index+batch_size]
            X_embeddings = embedding_layer(torch.tensor(batchX,
                                                        dtype=torch.int32).to(device=device))
            Y_embeddings = embedding_layer(torch.tensor(batchY,
                                                        dtype=torch.int32).to(device=device))
            print(f"X_embeddings: {X_embeddings.shape}, Y_embeddings: {Y_embeddings.shape}")
            # optimizer.zero_grad()
            # output = model(X_embeddings, cell_choice='rnn') # [batch_size, seq_length, input_size]
            # loss = criterion(output, Y_embeddings)
            # loss.backward()
            # optimizer.step()
            # print(f"Epoch {epoch} | Loss torch: {loss.item():.6f}")
    
    print(f"computational time: {time.time()-start_time}")




    # Questions:
    '''
    1. Does the input have to be of random length
    2. Does padding have to be for random steps in every sample?
    3. If input and padding length are random, do we need to make the input lengths consistent? how to make them consistent?
    '''