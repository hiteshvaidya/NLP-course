import torch
import torch.nn as nn
import numpy as np
import time
import pickle as pkl
from tqdm import tqdm
from loguru import logger
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse

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
        self.W_x = torch.nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_h = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
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
        self.W_z = torch.nn.Parameter(torch.randn(input_size, 
                                                  hidden_size) * 0.01)
        self.W_r = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.W_h = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.U_z = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.01)
        self.U_r = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.01)
        self.U_h = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.01)
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
        self.W_f = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.W_i = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.W_c = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.W_o = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.U_f = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.01)
        self.U_i = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.01)
        self.U_c = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.01)
        self.U_o = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.01)
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
        return h_t, c_t
    
class mLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_i = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.W_m = torch.nn.Parameter(torch.randn(input_size,
                                                  input_size) * 0.01)
        self.U_i = torch.nn.Parameter(torch.randn(hidden_size, 
                                                  hidden_size) * 0.01)
        self.U_m = torch.nn.Parameter(torch.randn(hidden_size,
                                                  input_size) * 0.01)
        self.W_f = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.U_f = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.01)
        self.W_o = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.U_o = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.01)
        self.W_c = torch.nn.Parameter(torch.randn(input_size, 
                                                  hidden_size) * 0.01)
        self.U_c = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.01)

        self.b_m = torch.nn.Parameter(torch.zeros(input_size))
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev, c_prev):
        m_t = x_t @ self.W_m + h_prev @ self.U_m + self.b_m
        x_tilde_t = m_t * x_t
        i_t = torch.sigmoid(x_tilde_t @ self.W_i + h_prev @ self.U_i + self.b_i)
        f_t = torch.sigmoid(x_tilde_t @ self.W_f + h_prev @ self.U_f + self.b_f)
        o_t = torch.sigmoid(x_tilde_t @ self.W_o + h_prev @ self.U_o + self.b_o)
        c_tilde_t = torch.tanh(x_tilde_t @ self.W_c + h_prev @ self.U_c + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t
    
class mGRU(torch.nn.Module):
    """
    source: https://arxiv.org/pdf/1907.00455
    """
    def __init__(self, input_size, hidden_size):
        super(mGRU, self).__init__()
        self.hidden_size = hidden_size
        self.W_m = torch.nn.Parameter(torch.randn(input_size,
                                                  input_size) * 0.01)
        self.W_z = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.W_r = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.W_h = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.01)
        self.U_m = torch.nn.Parameter(torch.randn(hidden_size, 
                                                  input_size) * 0.01)
        self.U_z = torch.nn.Parameter(torch.randn(hidden_size, 
                                                  hidden_size) * 0.01)
        self.U_r = torch.nn.Parameter(torch.randn(hidden_size, 
                                                  hidden_size) * 0.01)
        self.U_h = torch.nn.Parameter(torch.randn(hidden_size, 
                                                  hidden_size) * 0.01)
        
        self.b_m = torch.nn.Parameter(torch.zeros(input_size))
        self.b_z = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_r = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        m_t = x_t @ self.W_m + h_prev @ self.U_m + self.b_m
        x_tilde_t = m_t * x_t
        z_t = torch.sigmoid(x_tilde_t @ self.W_z + h_prev @ self.U_z + self.b_z)
        r_t = torch.sigmoid(x_tilde_t @ self.W_r + h_prev @ self.U_r + self.b_r)
        h_tilde_t = torch.tanh(x_tilde_t @ self.W_h + (r_t * h_prev) @ self.U_h + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde_t
        return h_t

def plot_loss(losses, title, path):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(path)
    plt.close()

class RNN(torch.nn.Module):
    """RNN network
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, 
                 device, model_choice='rnn'): # 
        super(RNN, self).__init__()

        self.device = device

        self.embedding_layer = nn.Embedding(vocab_size, 
                                            embedding_size).to(device)

        self.hidden_size = hidden_size
        # self.cell = None
        if model_choice == 'gru':
            self.cell = GRUCell(embedding_size, hidden_size)
        elif model_choice == 'lstm':
            self.cell = LSTMCell(embedding_size, hidden_size)
        elif model_choice == 'mlstm':
            self.cell = mLSTMCell(embedding_size, hidden_size)
        elif model_choice == 'mgru':
            self.cell = mGRU(embedding_size, hidden_size)
        else:
            self.cell = RNNCell(embedding_size, hidden_size)
        self.cell.to(device)
        
        # output dimension of W_out should be equal to input size for copy task
        # self.W_out = torch.nn.Parameter(torch.randn(hidden_size, vocab_size)).to(device)
        # self.b_out = torch.nn.Parameter(torch.zeros(vocab_size)).to(device)
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        # self.cell.to(device)
        self.to(device)

    # def parameters(self):
    #     params = list(super(RNN, self).parameters())
    #     if self.cell is not None:
    #         params.extend(list(self.cell.parameters()))
    #     for param in params:
    #         print(param.shape)
    #     return params

    def forward(self, X):
        """forward pass through the RNN

        Args:
            X (float): [batch_size, seq_len, input_size]
            cell_choice (str): rnn/gru
        """
        X = self.embedding_layer(X)
        batch_size, seq_len, _ = X.shape
        # initial hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=X.device)
        cell_state = None
        if isinstance(self.cell, LSTMCell) or isinstance(self.cell, mLSTMCell):
            cell_state = torch.zeros(batch_size, self.hidden_size, device=X.device)
        outputs = []
        for t in range(seq_len):
            x_t = X[:, t, :]    # [batch_size, input_size]
            if cell_state is not None:
                h, cell_state = self.cell(x_t, h, cell_state)   # [batch_size, hidden_size]
            else:
                h = self.cell(x_t, h)
            # if cell_choice == 'gru':
            #     h = self.gru_cell(x_t, h)
            # elif cell_choice == 'lstm':
            #     h, cell_state = self.lstm_cell(x_t, h, cell_state)
            # elif cell_choice == 'mlstm':
            #     h, cell_state = self.mlstm_cell(x_t, h, cell_state)
            # elif cell_choice == 'mgru':
            #     h = self.mgru_cell(x_t, h)
            # else:
            #     h = self.rnn_cell(x_t, h)
            
            # Project hidden -> input_size
            # out_t = h @ self.W_out + self.b_out
            logits = self.out_layer(h)
            # output = self.softmax(output)
            outputs.append(logits.unsqueeze(1))  # shape [batch_size, 1, input_size]
        # concatenate across time
        # if cell_state is not None:
        #     return torch.cat(outputs, dim=1), cell_state    # [batch_size, seq_length, input_size]
        # else:
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
def generate_sequences(seq_length, padding, vocabulary, 
                       delimiter, unknown, output_len):
    input_seq = []
    output = []
    for index in range(seq_length):
        input_seq.append(np.random.choice(vocabulary))
    output = input_seq.copy()
    for index in range(padding):
        input_seq.append(delimiter)
    output_padding = len(input_seq)
    for index in range(output_len):
        input_seq.append(unknown)
    output = output_padding * [unknown] + output[:output_len]
    return input_seq, output

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='mps, cuda, cpu')
    parser.add_argument('-l', '--log', type=str, default='rnn.log', 
                        help='filename.log')
    parser.add_argument('-iseq', '--train_seq_len', type=int, default=100, 
                        help='sequence length of train set')
    parser.add_argument('-tseq', '--test_seq_len', type=int, default=100, 
                        help='sequence length of test set')
    parser.add_argument('-p', '--padding', type=int, default=10, 
                        help='padding')
    parser.add_argument('-oseq', '--train_output_len', type=int, default=50, 
                        help='output length')
    parser.add_argument('-toseq', '--test_output_len', type=int, default=50, 
                        help='output length')
    parser.add_argument('-b', '--batch_size', type=int, default=32, 
                        help='batch size')
    parser.add_argument('-em', '--embedding_size', type=int, default=11, 
                        help='embedding size')
    parser.add_argument('-hs', '--hidden_size', type=int, default=128, 
                        help='hidden size')
    parser.add_argument('-c', '--cell', type=str, default='rnn', 
                        help='rnn/gru/lstm/mgru/mlstm')
    parser.add_argument('-ep', '--epochs', type=int, default=10, 
                        help='epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, 
                        help='learning rate')
    parser.add_argument('-s', '--seed', type=int, default=42, 
                        help='seed')
    parser.add_argument('-trs', '--train_samples', type=int, default=10000, 
                        help='train samples')
    parser.add_argument('-ts', '--test_samples', type=int, default=1000, 
                        help='test samples')
    args = parser.parse_args()

    # Configure the logger
    logger.add(args.log, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

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

    padding = args.padding  # repeat delimiter for how many time steps
    batch_size = args.batch_size
    input_size = len(vocabulary)
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    n_epochs = args.epochs
    lr = args.learning_rate
    train_samples = args.train_samples
    test_samples = args.test_samples

    np.random.seed(args.seed)

    train_size = args.train_seq_len
    test_size = args.test_seq_len
    train_output_len = args.train_output_len
    test_output_len = args.test_output_len

    # Generate train samples for copy task
    X_train = []
    Y_train = []
    tqdm.write(f"Generating {train_samples} train samples...")
    for index in tqdm(range(train_samples)):
        input_seq, output = generate_sequences(train_size, padding=padding, vocabulary=idx_vocab, delimiter=char2idx[delimiter], unknown=char2idx[unknown], output_len=train_output_len)
        X_train.append(input_seq)
        Y_train.append(output)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # pkl.dump((X_train, Y_train), open(f'copyTask_data_N{train_samples}_T{seq_length[0]}_P{padding[0]}_O{train_output_len}.pkl', 'wb'))

    # Generate test samples for copy task
    X_test = []
    Y_test = []
    tqdm.write(f"Generating {test_samples} test samples...")
    for index in tqdm(range(test_samples)):
        input_seq, output = generate_sequences(test_size, padding=padding, vocabulary=idx_vocab, delimiter=char2idx[delimiter], unknown=char2idx[unknown], output_len=test_output_len)
        X_test.append(input_seq)
        Y_test.append(output)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    # pkl.dump((X_test, Y_test), open(f'copyTask_data_N{train_samples}_T{seq_length[0]}_P{padding[0]}_O{test_output_len}.pkl', 'wb'))

    # (X_train, Y_train) = pkl.load(open('copyTask_data_N10000_T100_P10_O50.pkl', 'rb'))
    logger.info(f"Loaded train data with shape: {X_train.shape}, {Y_train.shape}")
    # (X_test, Y_test) = pkl.load(open('copyTask_data_N10000_T500_P50_O100.pkl', 'rb'))
    logger.info(f"Loaded test data with shape: {X_test.shape}, {Y_test.shape}")
    
    # Model
    model = RNN(len(vocabulary), embedding_size, 
                hidden_size, device, args.cell)
    for name, param in model.named_parameters():
        print(name, param.size())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Vocabulary: {vocabulary}")
    logger.info(f"Unknown: {unknown}, Delimiter: {delimiter}")
    logger.info(f"sequence length: {train_size}, padding: {padding}, output length: {train_output_len}")
    logger.info(f"batch size: {batch_size}, input size: {input_size}, embedding size: {embedding_size}, hidden size: {hidden_size}")
    logger.info(f"epochs: {n_epochs}, lr: {lr}")
    logger.info(f"train samples: {train_samples}, test samples: {test_samples}")
    
    train_losses = []
    train_accuracies = []
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        cell_state = torch.zeros(batch_size, hidden_size, device=device)
        # shuffle training data
        train_indices = torch.randperm(train_samples)
        tqdm.write("Training batches...")
        for index in tqdm(range(0, train_samples, batch_size)):
            batch_indices = train_indices[index:index+batch_size]
            batchX = torch.tensor(X_train[batch_indices], 
                                 dtype=torch.long).to(device)
            batchY = torch.tensor(Y_train[batch_indices]).to(device)
            current_batch_size = len(batch_indices)
            
            Y_one_hot = F.one_hot(batchY, input_size).float()
            optimizer.zero_grad()
            output = model(batchX) # [batch_size, seq_length, vocab_size]
            
            # Calculate loss at each time step
            time_step_losses = []
            for t in range(output.shape[1]):  # Iterate over sequence length
                time_step_output = output[:, t, :]  # [batch_size, vocab_size]
                time_step_target = Y_one_hot[:, t, :]  # [batch_size, vocab_size]
                time_step_loss = criterion(time_step_output, time_step_target)
                time_step_losses.append(time_step_loss)

            # Average time-step losses
            loss = torch.mean(torch.stack(time_step_losses))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item() * current_batch_size
            
            # Calculate accuracy at each time step and average
            time_step_accuracies = []
            for t in range(output.shape[1]):
                time_step_output = output[:, t, :]
                time_step_target = Y_one_hot[:, t, :]
                accuracy = (torch.argmax(time_step_output, dim=-1) == torch.argmax(time_step_target, dim=-1)).float().mean()
                time_step_accuracies.append(accuracy)
            train_accuracy += torch.mean(torch.stack(time_step_accuracies)).item() * current_batch_size

            # output = output[:, -train_output_len:, :]
            # Y_one_hot = Y_one_hot[:, -train_output_len:, :]
            # # predictions = [Batch, Time, Output]
            # train_accuracy += (torch.argmax(output, dim=-1) == torch.argmax(Y_one_hot, dim=-1)).sum().item()
        train_loss /= train_samples
        train_losses.append(train_loss)
        train_accuracy /= train_samples
        train_accuracies.append(train_accuracy)
        # print(f"Epoch {epoch} | Training, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # Log train metrics
        logger.info(f"Epoch {epoch} | Loss torch: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    # print(f"train time: {time.time()-start_time}")
    logger.info(f"train time: {time.time()-start_time}")
    
    # Test the model
    tqdm.write("Testing model...")
    test_loss = 0.0
    test_accuracy = 0.0
    start_time = time.time()
    for index in range(0, test_samples, batch_size):
        batchX = torch.tensor(X_test[index:index+batch_size]).to(device=device)
        batchY = torch.tensor(Y_test[index:index+batch_size]).to(device=device)
        current_batch_size = len(batchX)

        # If test data has more time steps than train data
        # Then we split the test data into multiple folds 
        # of the same size as the train data
        Y_one_hot = F.one_hot(batchY, input_size).float()
        predictions = torch.zeros_like(Y_one_hot)
        if test_size > train_size:
            folds = test_size // train_size
            for f in range(folds):
                X_fold = batchX[:, f*train_size:(f+1)*train_size]
                Y_fold = batchY[:, f*train_size:(f+1)*train_size]
                output = model(X_fold) # [batch_size, seq_length, input_size]
                predictions[:, f*train_size:(f+1)*train_size, :] = output

            # Calculate loss at each time step
            time_step_losses = []
            for t in range(Y_one_hot.shape[1]):  # Iterate over sequence length
                time_step_output = predictions[:, t, :]  # [batch_size, vocab_size]
                time_step_target = Y_one_hot[:, t, :]  # [batch_size, vocab_size]
                time_step_loss = criterion(time_step_output, time_step_target)
                time_step_losses.append(time_step_loss)
            # Average time-step losses
            loss = torch.mean(torch.stack(time_step_losses))
            test_loss += loss.item() * current_batch_size
            
            # Calculate accuracy at each time step and average
            time_step_accuracies = []
            for t in range(predictions.shape[1]):
                time_step_output = predictions[:, t, :]
                time_step_target = Y_one_hot[:, t, :]
                accuracy = (torch.argmax(time_step_output, dim=-1) == torch.argmax(time_step_target, dim=-1)).float().mean()
                time_step_accuracies.append(accuracy)
            train_accuracy += torch.mean(torch.stack(time_step_accuracies)).item() * current_batch_size

            # loss = criterion(predictions, Y_one_hot)
            # predictions = predictions[:, -test_output_len:, :]
            # Y_one_hot = Y_one_hot[:, -test_output_len:, :]
            # accuracy = (torch.argmax(output, dim=-1) == torch.argmax(Y_one_hot, dim=-1)).sum().item()
            # test_loss += loss.item() * current_batch_size
            # test_accuracy += accuracy
    test_loss /= test_samples
    test_accuracy /= test_samples
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    # print(f"test time: {time.time()-start_time}")
    logger.info(f"test time: {time.time()-start_time}")

    # Save model
    torch.save(model, 'rnn.pth')

    # Questions:
    '''
    1. Does the input have to be of random length
    2. Does padding have to be for random steps in every sample?
    3. If input and padding length are random, do we need to make the input lengths consistent? how to make them consistent?
    '''