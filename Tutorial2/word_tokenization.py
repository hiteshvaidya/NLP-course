import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import random
import pandas as pd


tokenization_table = {'model':[], 'method': [], 'vocab_size':[], 'sequence_len':[]}

metric_log = {'model':[], 'seed': [], 'layers':[], 'activation': [], 'optimizer': [], 'learning_rate': [], 'test_accuracy': [], 'test_loss': [], 'precision': [], 'recall': []}

visited = []

def set_new_seed(trial_number):
    """set new seed for every trial

    Args:
        trial_number (int): trial number

    Returns:
        [None]: None
    """
    random_seed = random.randint(0, 2**32-1)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    return random_seed

# -------------------------------
# Original MLP Class Definition
# -------------------------------
class MLP(object):
    def __init__(self, layers, activation, device=None):
        """
        layers: list containing dimensions of all layers
        activation: str, indicating choice of activation function
        device: str or None, either 'cpu' or 'gpu' or None.
        """
        self.layers = layers
        self.device = device
        # list of all weights and biases
        self.W = []
        self.b = []

        # Initialize weights and biases for each layer
        for l in range(len(self.layers[1:])):
            if l == 0:
                continue
            W = tf.Variable(tf.random.normal([
                            self.layers[l-1], self.layers[l]], 
                            stddev=0.1))
            self.W.append(W)
            b = tf.Variable(tf.zeros([1, self.layers[l]]))
            self.b.append(b)

        # List of variables to update during backpropagation
        self.variables = self.W + self.b

        # activation function
        if activation == 'ReLU':
            self.activation = tf.nn.relu()
        elif activation == 'Tanh':
            self.activation = tf.nn.tanh()
        elif activation == 'LeakyReLU':
            self.activation = tf.nn.leaky_relu()

    def forward(self, X):
        """
        Forward pass.
        X: Tensor, inputs.
        """
        if self.device is not None:
            # use '/GPU:0' instead of 'gpu:0' for using gpu on mac
            with tf.device('/GPU:0' if self.device == 'gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            self.y = self.compute_output(X)
        return self.y

    def loss(self, y_pred, y_true):
        """
        Computes the loss between predicted and true outputs.
        y_pred: Tensor of shape (batch_size, size_output)
        y_true: Tensor of shape (batch_size, size_output)
        """
        y_true_tf = tf.cast(y_true, dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss_x = cce(y_true_tf, y_pred_tf)
        return loss_x

    def backward(self, X_train, y_train):
        """
        Backward pass: compute gradients of the loss with respect to the variables.
        """
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
        grads = tape.gradient(current_loss, self.variables)
        return grads

    def compute_output(self, X):
        """
        Custom method to compute the output tensor during the forward pass.
        """
        # Cast X to float32
        z = tf.cast(X, dtype=tf.float32)
        for l in range(len(self.layers)-1):
            h = tf.matmul(z, self.W[l]) + self.b[l]
            z = self.activation(h)
        
        # Output layer (logits)
        output = tf.matmul(z, self.W[-1]) + self.b[-1]
        return output

# -------------------------------
# Character-Level Tokenizer and Preprocessing Functions
# -------------------------------
def char_level_tokenizer(texts, char_level, num_words=None):
    """
    Create and fit a character-level tokenizer.

    Args:
        texts (list of str): List of texts.
        num_words (int or None): Maximum number of tokens to keep.

    Returns:
        tokenizer: A fitted Tokenizer instance.
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, char_level=char_level, lower=True)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def texts_to_bow(tokenizer, texts):
    """
    Convert texts to a bag-of-characters representation.

    Args:
        tokenizer: A fitted character-level Tokenizer.
        texts (list of str): List of texts.

    Returns:
        Numpy array representing the binary bag-of-characters for each text.
    """
    # texts_to_matrix with mode 'binary' produces a fixed-length binary vector per text.
    matrix = tokenizer.texts_to_matrix(texts, mode='binary')
    return matrix

def one_hot_encode(labels, num_classes=2):
    """
    Convert numeric labels to one-hot encoded vectors.
    """
    return np.eye(num_classes)[labels]

# -------------------------------
# Load and Prepare the IMDB Dataset
# -------------------------------
print("Loading IMDB dataset...")
# Load the IMDB reviews dataset with the 'as_supervised' flag so that we get (text, label) pairs.
(ds_train, ds_test), ds_info = tfds.load('imdb_reviews',
                                           split=['train', 'test'],
                                           as_supervised=True,
                                           with_info=True)

# Convert training dataset to lists.
train_texts = []
train_labels = []
for text, label in tfds.as_numpy(ds_train):
    # Decode byte strings to utf-8 strings.
    train_texts.append(text.decode('utf-8'))
    train_labels.append(label)
train_labels = np.array(train_labels)

# Create a validation set from the training data (20% for validation).
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42)

# Convert test dataset to lists.
test_texts = []
test_labels = []
for text, label in tfds.as_numpy(ds_test):
    test_texts.append(text.decode('utf-8'))
    test_labels.append(label)
test_labels = np.array(test_labels)

print(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}, Test samples: {len(test_texts)}")

# -------------------------------
# Preprocessing: Tokenization and Vectorization
# -------------------------------
# Build the word-level tokenizer on the training texts.
tokenizer = char_level_tokenizer(train_texts, char_level=False)
print("Tokenizer vocabulary size:", len(tokenizer.word_index) + 1)

# Convert texts to bag-of-words representation.
X_train = texts_to_bow(tokenizer, train_texts)
X_val   = texts_to_bow(tokenizer, val_texts)
X_test  = texts_to_bow(tokenizer, test_texts)

# record tokenization details
tokenization_table['model'] = 'mlp'
tokenization_table['method'] = 'word'
tokenization_table['vocab_size'] = len(tokenizer.word_index) + 1
tokenization_table['sequence_len'] = X_train.shape[1]


# Convert labels to one-hot encoding.
y_train = one_hot_encode(train_labels)
y_val   = one_hot_encode(val_labels)
y_test  = one_hot_encode(test_labels)

# -------------------------------
# Model Setup
# -------------------------------
# List of dimensions in all the layers of the network
# The input size is determined by the dimension of the bag-of-characters vector.
hidden_layers = [1, 2, 3]
hidden_sizes = [128, 256, 512]
output_size = 2

# activation
activations = ['ReLU', 'Tanh', 'LeakyReLU']

# learning rate
learning_rates = [0.001, 0.0005, 0.0001]

# batch size
batch_sizes = [32, 64, 128]
# optimizer
optims = ['SGD', 'Adam', 'RMSProp']

# Instantiate the MLP model.
model = MLP([X_train.shape[1], 128, 64, 32, 2], activation, device='gpu')

# learning rate
learning_rate = 0.01

# optimizer
optimizer = {}
optimizer['Adam'] = tf.optimizers.Adam(learning_rate=learning_rate)
optimizer['SGD'] = tf.optimizers.SGD(learning_rate=learning_rate)
optimizer['RMSProp'] = tf.optimizers.RMSprop(learning_rate=learning_rate)
optim_choice = 'SGD'

# -------------------------------
# Training Parameters and Loop
# -------------------------------
epochs = 10

def generate_params():
    while True:
        n_hidden = random.choice(hidden_layers)
        activation = random.choice(activations)
        lr = random.choice(learning_rates)
        optim = random.choice(optims)
        batch_size = random.choice(batch_sizes)
        parameters = [n_hidden, activation, lr, optim, batch_size]
        if not parameters in visited:
            return parameters

trials = 60

while trials >=0:
    random_seed = set_new_seed(trials)

    parameters = generate_params()
    nn_layers = [X_train.shape[1]] + hidden_sizes[:parameters[0]] + [2]
    activation = parameters[1]
    lr = parameters[2]
    optim = parameters[3]
    if optim == 'Adam':
        optimizer = tf.optimizers.Adam(learning_rate=lr)
    elif optim == 'SGD':
        optimizer = tf.optimizers.SGD(learning_rate=lr)
    elif optim == 'RMSProp':
        optimizer = tf.optimizers.RMSprop(learning_rate=lr)
    batch_size = parameters[-1]

    best_test_acc = -100
    best_precision = None
    best_recall = None
    best_loss = None
    for i in range(3):
        # Instantiate the MLP model.
        model = MLP(nn_layers, activation, device='gpu')

        num_batches = int(np.ceil(X_train.shape[0] / batch_size))

        print(f"\nTraining for trial {60-trials+1}...\n")

        for epoch in range(epochs):
            # Shuffle training data at the start of each epoch.
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0
            for i in range(num_batches):
                start = i * batch_size
                end = min((i+1) * batch_size, X_train.shape[0])
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # Compute gradients and update weights.
                # with tf.GradientTape() as tape:
                #     predictions = model.forward(X_batch)
                #     loss_value = model.loss(predictions, y_batch)
                # grads = tape.gradient(loss_value, model.variables)
                predictions = model.forward(X_batch)
                loss_value = model.loss(predictions, y_batch)
                grads = model.backward(X_batch, y_batch)
                optimizer.apply_gradients(zip(grads, model.variables))
                epoch_loss += loss_value.numpy() * (end - start)

            epoch_loss /= X_train.shape[0]

            # Evaluate on validation set.
            val_logits = model.forward(X_val)
            val_loss = model.loss(val_logits, y_val).numpy()
            val_preds = np.argmax(val_logits.numpy(), axis=1)
            true_val = np.argmax(y_val, axis=1)
            accuracy = np.mean(val_preds == true_val)
            precision = precision_score(true_val, val_preds)
            recall = recall_score(true_val, val_preds)

            print(f"Epoch {epoch+1:02d} | Training Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # -------------------------------
        # Final Evaluation on Test Set
        # -------------------------------
        num_batches = int(np.ceil(X_test.shape[0] / batch_size))
        test_precision = 0.0
        test_recall = 0.0
        test_accuracy = 0.0
        true_test = []
        test_preds = []
        print("\nEvaluating on test set...")
        for b in range(num_batches):
            start = b*batch_size
            end = min((b+1)*batch_size, X_test.shape[0])
            X_batch = X_test[start: end]
            test_logits = tf.nn.softmax(model.forward(X_batch), axis=1)
            y_batch = y_test[start: end]
            test_loss = model.loss(test_logits, y_batch).numpy()
            predictions = np.argmax(test_logits.numpy(), axis=1)
            test_preds.extend(predictions)
            labels = np.argmax(y_batch, axis=1)
            true_test.extend(labels)

        test_accuracy = np.mean(np.array(test_preds) == np.array(true_test))
        test_precision = precision_score(true_test, test_preds)
        test_recall = recall_score(true_test, test_preds)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_precision = test_precision
            best_recall = test_recall
            best_loss = test_loss

        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f} | "
            f"Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f}")

    metric_log['model'].append('mlp_word_tokens')
    metric_log['seed'].append(random_seed)
    metric_log['layers'].append(parameters[0])
    metric_log['activation'].append(activation)
    metric_log['optimizer'].append(optim)
    metric_log['learning_rate'].append(lr)
    metric_log['test_accuracy'].append(round(best_test_acc, 4))
    metric_log['test_loss'].append(round(best_loss, 4))
    metric_log['precision'].append(round(best_precision, 4))
    metric_log['recall'].append(round(best_recall, 4))


tokenization_df = pd.DataFrame(tokenization_table)
tokenization_df.to_csv('word_tokenization_details.csv', index=False)

print("tokenization details:")
print(tokenization_df.head())

exp_df = pd.DataFrame(metric_log)
exp_df.to_csv('experiments_log.csv', index=False)
print("experiments details:")
print(exp_df.head())