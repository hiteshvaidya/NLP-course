# Assignment 3

Building Recurrent Neural Networks. </br>
Types of RNN cells: </br>
- RNN
- LSTM
- GRU
- mLSTM
- mGRU

## Requirements
[requirements.txt](./requirements.txt) has all the required libraries. </br>
`pip install -r requirements.txt`


## Data Generation
`Vocabulary = ['a','x','c','r','y','w','b','t','o']` </br>
`delimiter character = '$', unknown character = ' '` </br>

sample input:  </br>
```
['b', 'r', 't', 'y', 'b', 'c', 'b', 't', 'y', 'r', 't', 't', 'c',
       'w', 'y', 'x', 't', 'w', 'x', 'y', 'a', 'w', 'o', 'a', 'c', 'b',
       'r', 'o', 'c', 'y', 'c', 'b', 'y', 'o', 'b', 'x', 'r', 'o', 'x',
       'o', 'y', 'x', 'r', 'b', 't', 'c', 'a', 'r', 'x', 't', 'r', 'x',
       'w', 'w', 'r', 'w', 'x', 'x', 'r', 't', 'b', 'o', 't', 'y', 'x',
       'y', 't', 'o', 'o', 'a', 'o', 'b', 'o', 't', 'a', 't', 't', 'c',
       'a', 't', 'c', 'c', 'a', 'y', 'b', 'o', 'b', 'o', 't', 'x', 'a',
       'b', 'b', 't', 'y', 'c', 't', 'w', 'c', 'a', '$', '$', '$', '$',
       '$', '$', '$', '$', '$', '$', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ']
```

sample output: </br>
```
[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', 'b', 'r', 't', 'y', 'b', 'c', 'b',
       't', 'y', 'r', 't', 't', 'c', 'w', 'y', 'x', 't', 'w', 'x', 'y',
       'a', 'w', 'o', 'a', 'c', 'b', 'r', 'o', 'c', 'y', 'c', 'b', 'y',
       'o', 'b', 'x', 'r', 'o', 'x', 'o', 'y', 'x', 'r', 'b', 't', 'c',
       'a', 'r', 'x', 't']
```

The samples generated as shown above are then converted to integers using a lookup table. 
input sample converted from character to index: </br>
```
['b', 'r', 't', 'y', 'b', 'c', 'b', 't', 'y', 'r', 't', 't', 'c',
       'w', 'y', 'x', 't', 'w', 'x', 'y', 'a', 'w', 'o', 'a', 'c', 'b',
       'r', 'o', 'c', 'y', 'c', 'b', 'y', 'o', 'b', 'x', 'r', 'o', 'x',
       'o', 'y', 'x', 'r', 'b', 't', 'c', 'a', 'r', 'x', 't', 'r', 'x',
       'w', 'w', 'r', 'w', 'x', 'x', 'r', 't', 'b', 'o', 't', 'y', 'x',
       'y', 't', 'o', 'o', 'a', 'o', 'b', 'o', 't', 'a', 't', 't', 'c',
       'a', 't', 'c', 'c', 'a', 'y', 'b', 'o', 'b', 'o', 't', 'x', 'a',
       'b', 'b', 't', 'y', 'c', 't', 'w', 'c', 'a', '$', '$', '$', '$',
       '$', '$', '$', '$', '$', '$', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
       ' ', ' ', ' ', ' ']
```

## Training/Testing
The program uses command line arguments </br>
| Argument                 | Variable Name       | Description                                 |
|--------------------------|---------------------|---------------------------------------------|
| `-d`, `--device`         | `device`            | Specifies the device to use: 'mps', 'cuda', or 'cpu'. Default is 'cpu'. |
| `-l`, `--log`            | `log`               | Filename for logging. Default is 'rnn.log'. |
| `-iseq`, `--train_seq_len` | `train_seq_len`     | Sequence length of the training set. Default is 100. |
| `-tseq`, `--test_seq_len` | `test_seq_len`      | Sequence length of the test set. Default is 100. |
| `-p`, `--padding`        | `padding`           | Padding size. Default is 10.                |
| `-oseq`, `--train_output_len` | `train_output_len` | Output length for the training set. Default is 50. |
| `-toseq`, `--test_output_len` | `test_output_len`  | Output length for the test set. Default is 50. |
| `-b`, `--batch_size`     | `batch_size`        | Batch size. Default is 32.                  |
| `-em`, `--embedding_size` | `embedding_size`    | Size of the embeddings. Default is 11.      |
| `-hs`, `--hidden_size`   | `hidden_size`       | Size of the hidden layers. Default is 128.  |
| `-c`, `--cell`           | `cell`              | Type of RNN cell to use: 'rnn', 'gru', 'lstm', 'mgru', or 'mlstm'. Default is 'rnn'. |
| `-ep`, `--epochs`        | `epochs`            | Number of training epochs. Default is 10.   |
| `-lr`, `--learning_rate` | `learning_rate`     | Learning rate for the optimizer. Default is 0.1. |
| `-s`, `--seed`           | `seed`              | Random seed for reproducibility. Default is 42. |
| `-trs`, `--train_samples` | `train_samples`     | Number of training samples. Default is 10,000. |
| `-ts`, `--test_samples`  | `test_samples`      | Number of test samples. Default is 1,000.   |

sample input: </br>
`python rnn.py -d cuda -l lstm_100_E32_H256_lr0.001_N50k_trial1 -iseq 100 -tseq 500 -p 3 -oseq 100 -toseq 200 -b 32 -em 16 -hs 256 -c lstm -ep 15 -lr 0.001 -s 42 -trs 50000 -ts 10000` </br>

The [run.sh](./run.sh) specifies a list of all the run commands that I used for testing the code

**PS: to utilize gpu on mac, please specify `-d mps` in the command line arguments. If the pytorch in virtual environment is installed for metal, it will utilize gpu on mac**

### Logs
All the logs are saved as per the name provided along with `-l or --log` in the command line argument. Any experiment saves the respective model (.pth), runtime log, plots for loss and accuracy. To load a saved trained model (`.pth`), you can comment [line 557 of rnn.py](rnn.py#L557) and load the existing model at [line 561](rnn.py#L561) by providing the correct relative path.

