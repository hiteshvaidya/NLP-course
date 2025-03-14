import numpy as np
import pickle as pkl
from tqdm import tqdm

def generate_sequences(seq_length, padding, vocabulary, 
                       delimiter, unknown, output_len):
    input = []
    output = []
    for index in range(seq_length):
        input.append(np.random.choice(vocabulary))
    output = input.copy()
    for index in range(padding):
        input.append(delimiter)
    output_padding = len(input)
    for index in range(output_len):
        input.append(unknown)
    output = output_padding * [unknown] + output[:output_len]
    return input, output


if __name__ == "__main__":
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
    batch_size = 32
    input_size = len(vocabulary)
    hidden_size = 128
    n_epochs = 10
    lr = 0.01
    n_samples = 10000

    np.random.seed(42)

    X_train = []
    Y_train = []
    tqdm.write(f"Generating {n_samples} train samples...")
    for index in tqdm(range(n_samples)):
        input, output = generate_sequences(seq_length[0], padding=padding[0], vocabulary=idx_vocab, delimiter=char2idx[delimiter], unknown=char2idx[unknown], output_len=output_len[0])
        X_train.append(input)
        Y_train.append(output)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    pkl.dump((X_train, Y_train), open('copyTask_data_N10000_T100_P10_O50.pkl', 'wb'))

    X_train = []
    Y_train = []
    tqdm.write(f"Generating {n_samples} test samples...")
    for index in tqdm(range(n_samples)):
        input, output = generate_sequences(seq_length[2], padding=padding[2], vocabulary=idx_vocab, delimiter=char2idx[delimiter], unknown=char2idx[unknown], output_len=output_len[1])
        X_train.append(input)
        Y_train.append(output)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    pkl.dump((X_train, Y_train), open('copyTask_data_N10000_T500_P50_O100.pkl', 'wb'))