import torch
import torch.nn as nn
import numpy as np
import random
import utils.grid_utils as grid_utils
import math

# Based on: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# Reference paper: "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I.
# (2017). Attention is all you need. Advances in neural information processing systems, 30."

# This is the training script for experiment Full-Transformer-CountV2.
# Set TRAIN_MODEL to True to train the model, and False to evaluate a pre-trained model.
# RESUME_MODEL can be set to True to resume training from a previous training session.

np.set_printoptions(suppress=True)

RESUME_MODEL = False
TRAIN_MODEL = True

NUM_EPOCHS = 10000
vocab_size = 12      # 10 digits + EOW + EOS
device = 'cuda'
LR = 0.0002
num_heads = 1
train_batch_size = 50

EMB_DIM = 64

# ================================================== Data generation =================================================

# Only counts non-zero pixels
class MediumVaryingCountingTask():
    # update this task to reflect output as:
    #  [count 0 digit 1, count 0 digit 2, ..., <end of word>, count 1 digit 1, count 2 digit 2, ..., <end of word>, etc.]
    def __init__(self, num_px=99, max_grid_dim=10):
        self.num_px = num_px
        self.max_grid_dim = max_grid_dim

    def generateInputs(self, k, grid_dim=None):
        input_grids = []

        if grid_dim is None:
            grid_dim = np.random.choice(np.arange(1, self.max_grid_dim))

        for _ in range(k):
            tmp_grid = self._generateInput(grid_dim, grid_dim * grid_dim)
            input_grids.append(tmp_grid)

        random.shuffle(input_grids)
        return input_grids

    def _generateInput(self, grid_dim, mpt):
        return grid_utils.generateRandomPixels(max_pixels_total=mpt,
                                               max_pixels_per_color=self.num_px,
                                               grid_dim_min=grid_dim,
                                               grid_dim_max=grid_dim,
                                               sparsity=1.)

    def _generateOutput(self, input_grid):
        pixel_count = grid_utils.perColorPixelCountV2(input_grid)[1:]
        pixel_seq = []

        for pxc in pixel_count:
            if pxc < 10:
                str_count = "0%i" % pxc
            else:
                str_count = "%i" % pxc

            for char in str_count:
                pixel_seq.append(int(char))

        return np.array(pixel_seq)

    def generateOutputs(self, input_grids):
        output_grids = []

        for input_grid in input_grids:
            output_grids.append(self._generateOutput(input_grid))

        return np.array(output_grids)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(d_model=dim_model, dropout=dropout_p, max_len=20)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, tgt_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        #src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a square matrix where each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        return mask

def generate_random_data(n, task_instance, grid_dim=None):
    SOS_token = np.array([10])
    EOS_token = np.array([11])

    source = np.array(task_instance.generateInputs(n, grid_dim))
    target = task_instance.generateOutputs(source)

    data = []
    for idx in range(source.shape[0]):
        x = source[idx]

        tmp_x = np.concatenate((SOS_token, np.reshape(x, [-1]), EOS_token))

        y = target[idx]

        tmp_y = np.concatenate((SOS_token, np.reshape(y, [-1]), EOS_token))

        data.append([tmp_x, tmp_y])

    np.random.shuffle(data)

    return data

def batchify_data(data, batch_size=100, padding=False, padding_token=-1):
    batches = []
    for idx in range(0, len(data), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size <= len(data):
            # Here you would need to get the max length of the batch,
            # and normalize the length with the PAD token.
            if padding:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(data[idx : idx + batch_size])

    return batches

task_instance = MediumVaryingCountingTask()

# ================================================== Model training =================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(num_tokens=vocab_size, dim_model=EMB_DIM, num_heads=num_heads,
                    num_encoder_layers=1, num_decoder_layers=25, dropout_p=0.).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss()

def train_loop(model, opt, loss_fn):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.train()
    total_loss = 0

    train_data = generate_random_data(1000, task_instance)
    train_dataloader = batchify_data(train_data)

    for batch in train_dataloader:

        X = []
        y = []
        for b_idx in range(len(batch)):
            X.append(batch[b_idx][0])
            y.append(batch[b_idx][1])

        X = np.array(X)
        y = np.array(y)
        X, y = torch.tensor(X).to(device).long(), torch.tensor(y).to(device).long()

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(train_dataloader)

def fit(model, opt, loss_fn, epochs):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Used for plotting later on
    train_loss_list = []

    best_loss = np.inf
    print("==> Training and validating model")
    for epoch in range(epochs):
        train_loss = train_loop(model, opt, loss_fn)
        train_loss_list += [train_loss]

        print("Epoch #%i - loss = %.4f" % (epoch+1, train_loss))

        if len(train_loss_list) >= 20:
            mean_loss = np.mean(train_loss_list[-20:])

            if mean_loss < best_loss:
                best_loss = mean_loss
                print("==> Saving new best model!")
                torch.save(model.state_dict(), 'FullCounter-transformer.pt')

    return train_loss_list

if TRAIN_MODEL:
    train_loss_list = fit(model, opt, loss_fn, NUM_EPOCHS)
else:
    # load model
    model.load_state_dict(torch.load('FullCounter-transformer.pt'))
    model = model.double().to(device)

# ================================================== Model evaluation =================================================
model.eval()

def predict(model, input_sequence, max_length=18, SOS_token=10):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    model.eval()

    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device).long()

    input_sequence = torch.reshape(input_sequence, [1, -1])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device).double()

        # y_input shape = [1, seq_length]
        # input_sequence = always [1, 11] -- the input grid + SOS + EOS
        pred = model(input_sequence, y_input, tgt_mask)

        # pred shape = [seq_length, 1, vocab_size]
        next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop after fixed length
        if y_input.shape[1] >= 19:
            break

    return y_input.view(-1).tolist()

# Here we test some examples to observe how the model predicts
train_data = generate_random_data(10, task_instance, grid_dim=15)
train_dataloader = batchify_data(train_data, batch_size=10)

X = []
y = []
for b_idx in range(len(train_dataloader[0])):

    X.append(train_dataloader[0][b_idx][0])
    y.append(train_dataloader[0][b_idx][1])

X = np.array(X)
y = np.array(y)
X, y = torch.tensor(X).to(device).long(), torch.tensor(y).to(device).long()

# Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
# X shape = [10, 11]
# y shape = [10, 20]
y_input = y[:, :-1]
y_expected = y[:, 1:]

def accuracy(res_pred, tgts):
    tmp_acc = 0
    for idx in range(len(tgts)):
        if res_pred[idx] == tgts[idx]:
            tmp_acc += 1

    return float(tmp_acc) / float(len(tgts))

total_accuracy = 0
for idx, example in enumerate(X):
    result = predict(model, example)
    print(f"Example {idx}")
    print(f"Input: {example.view(-1).tolist()[1:-1]}")
    print(f"Predicted values: {result[1:]}")
    print("Ground truths: ", y_expected[idx, :-1].cpu().data.numpy())
    acc = accuracy(result[1:], y_expected[idx, :-1].cpu().data.numpy())
    total_accuracy += acc
    print()

total_accuracy /= len(X)
print("Total accuracy = ", total_accuracy)

# Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
# Get mask to mask out the next words
sequence_length = y_input.size(1)
tgt_mask = model.get_tgt_mask(sequence_length).to(device).double()

# Standard training except we pass in y_input and tgt_mask
pred = model(X, y_input, tgt_mask)

# Permute pred to have batch size first again
pred = pred.permute(1, 2, 0)
loss = loss_fn(pred, y_expected)

# pred shape = [batch_size, vocab_size, seq_length=19 rather than 18!]
# y_expected shape = [batch_size, 19 instead of 18!]
print("pred shape = ", pred.shape)
print("y expected shape = ", y_expected.shape)

print("pred = ", pred.cpu().data.numpy())
print("y expected = ", y_expected.cpu().data.numpy())

print("loss = ", loss.cpu().data.numpy())
