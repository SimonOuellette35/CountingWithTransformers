import torch
import torch.nn as nn
import numpy as np
import random
import utils.grid_utils as grid_utils
import math
import torch.nn.functional as F

# Based on: https://github.com/iibrahimli/universal_transformers/

#  4) Try adding positional encoding to the encoder side input as well
#  5) Try intermediate supervision: at each time step, must return the number of instnaces up to that
#   time step (== cell in the memory grid)
#  6) halting is done based on the state -- does that or can that include the current value of the memory cell that
#  was attended to (i.e. the SOS token?)? Also, what is the mechanism to increment the positional encoding to attend to
#  from iteration to iteration of the same decoding layer?
#   a) Implement a barebones, simplistic version of the decoder algorithm to count iteratively. Does it work?
#   b) Then, take that and make it iterative with ACT as in the UT?
#  7) Try learning only the modulo 10?

np.set_printoptions(suppress=True)

RESUME_MODEL = False
TRAIN_MODEL = True

NUM_EPOCHS = 10000
vocab_size = 12      # 10 digits + EOW + EOS
device = 'cuda'
num_heads = 1
train_batch_size = 50

EMB_DIM = 64

# Only counts non-zero pixels
class MediumCountingTask():
    # update this task to reflect output as:
    #  [count 0 digit 1, count 0 digit 2, ..., <end of word>, count 1 digit 1, count 2 digit 2, ..., <end of word>, etc.]
    def __init__(self, num_px=99, grid_dim=6):
        self.num_px = num_px
        self.grid_dim = grid_dim

    def generateInputs(self, k):
        input_grids = []
        for _ in range(k):
            tmp_grid = self._generateInput(self.grid_dim * self.grid_dim)
            input_grids.append(tmp_grid)

        random.shuffle(input_grids)
        return input_grids

    def _generateInput(self, mpt):
        return grid_utils.generateRandomPixels(max_pixels_total=mpt,
                                               max_pixels_per_color=self.num_px,
                                               grid_dim_min=self.grid_dim,
                                               grid_dim_max=self.grid_dim,
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

class UniversalTransformer(nn.Module):
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        dropout_p,
        encoding_max_timestep,
        decoding_max_timestep,
        halting_thresh=0.8
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        self.encoding_max_timestep = encoding_max_timestep
        self.decoding_max_timestep = decoding_max_timestep
        self.halting_thresh = halting_thresh

        self.timing_signal = self.gen_timing_signal(20, dim_model)

        self.enc_position_signal = self.gen_timing_signal(encoding_max_timestep, dim_model)
        self.dec_position_signal = self.gen_timing_signal(decoding_max_timestep, dim_model)

        # LAYERS
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.encoder_layer = nn.TransformerEncoderLayer(dim_model, num_heads, activation=nn.GELU(), batch_first=True)
        self.decoder_layer = nn.TransformerDecoderLayer(dim_model, num_heads, activation=nn.GELU(), batch_first=True)
        self.enc_halting_layer1 = nn.Linear(dim_model, 2048)
        self.enc_halting_layer2 = nn.Linear(2048, 1)
        self.dec_halting_layer1 = nn.Linear(dim_model, 2048)
        self.dec_halting_layer2 = nn.Linear(2048, 1)

        self.out = nn.Linear(dim_model, num_tokens)

    def gen_timing_signal(self, length, channels, min_timescale=1.0, max_timescale=1.0e4):
        """
        Generates a [1, length, channels] timing signal consisting of sinusoids
        Adapted from:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        """
        position = np.arange(length)
        num_timescales = channels // 2
        log_timescale_increment = (
                    math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * np.exp(
            np.arange(num_timescales).astype(float) * -log_timescale_increment)
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                        'constant', constant_values=[0.0, 0.0])
        signal = signal.reshape([1, length, channels])

        return torch.from_numpy(signal).type(torch.FloatTensor)

    def forward(self, src, tgt, tgt_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)

        memory, ponder_time = self.UT_encode(src)
        out = self.UT_decode(memory, tgt, tgt_mask=tgt_mask)
        out = self.out(out)

        return out

    def UT_encode(self, source):
        """
        Perform forward pass of the encoder.

        Args:
           source: Tensor of shape [batch_size, src_seq_len, embedding_dim]
           source_padding_mask: Mask of shape [batch_size, src_seq_len]

        Returns:
           Has shape [batch_size, src_seq_len, embedding_dim]
        """
        halting_probability = torch.zeros((*source.shape[:-1], 1), device=device)
        remainders = torch.zeros_like(halting_probability)
        n_updates = torch.zeros_like(halting_probability)
        ponder_time = torch.zeros_like(halting_probability)

        new_src = source.clone()

        for time_step in range(self.encoding_max_timestep):
            # Add timing signal
            #new_src = new_src + self.enc_position_signal[:, time_step, :].unsqueeze(1).repeat(1, source.shape[1], 1).type_as(source.data)

            #still_running = halting_probability < self.halting_thresh
            #tmp_src = self.enc_halting_layer1(new_src)
            #p = F.sigmoid(self.enc_halting_layer2(F.relu(tmp_src)))

            #new_halted = (halting_probability + p * still_running) > self.halting_thresh
            #ponder_time[~new_halted] += 1
            #still_running = (halting_probability + p * still_running) <= self.halting_thresh
            #halting_probability += p * still_running

            #remainders += new_halted * (1 - halting_probability)
            #halting_probability += new_halted * remainders

            #n_updates += still_running + new_halted
            #update_weights = p * still_running + new_halted * remainders

            new_src = self.encoder_layer(new_src)
            source = new_src + source

        return source, ponder_time

    def UT_decode(self, memory, target, tgt_mask):
        """
        Perform forward pass of the decoder.

        Args:
            memory: Has shape [batch_size, src_seq_len, embedding_dim]
            target: Has shape [batch_size, tgt_seq_len]
            target_mask: Has shape [tgt_seq_len, tgt_seq_len]
            memory_padding_mask: Has shape [batch_size, src_seq_len, embedding_dim]
            target_padding_mask: Has shape [batch_size, tgt_seq_len]

        Returns:
            Has shape [batch_size, tgt_seq_len, embedding_dim]
        """
        running_probability = torch.ones((*target.shape[:-1], 1), device=device)
        remainders = torch.zeros_like(running_probability)
        still_running = torch.ones_like(running_probability)
        new_target = target.clone()

        self.dec_ponder_time = torch.zeros_like(running_probability)

        # for time_step in range(self.decoding_max_timestep):
        #     # Add timing signal
        #     new_target = new_target + self.timing_signal[:, :target.shape[1], :].type_as(target.data)
        #     new_target = new_target + self.dec_position_signal[:, time_step, :].unsqueeze(1).repeat(1, target.shape[1], 1).type_as(target.data)
        #
        #     still_running = halting_probability < self.halting_thresh
        #     p = self.halting_layer(new_target)
        #     new_halted = (halting_probability + p * still_running) > self.halting_thresh
        #     still_running = (halting_probability + p * still_running) <= self.halting_thresh
        #     halting_probability += p * still_running
        #     remainders += new_halted * (1 - halting_probability)
        #     halting_probability += new_halted * remainders
        #     n_updates += still_running + new_halted
        #     update_weights = p * still_running + new_halted * remainders
        #     new_target = self.decoder_layer(new_target, memory, tgt_mask=tgt_mask)
        #     target = (new_target * update_weights) + (target * (1 - update_weights))
        #
        #     # update counter
        #     self.dec_ponder_time[~new_halted] += 1

        for time_step in range(self.decoding_max_timestep):
            # Add timing signal
            new_target = new_target + self.timing_signal[:, :target.shape[1], :].type_as(target.data)
            #new_target = new_target + self.dec_position_signal[:, time_step, :].unsqueeze(1).repeat(1, target.shape[1], 1).type_as(target.data)

            #tmp_target = self.dec_halting_layer1(new_target)
            #p = F.sigmoid(self.dec_halting_layer2(F.relu(tmp_target)))

            #running_probability = p * still_running
            #still_running = running_probability > (1 - self.halting_thresh)
            #new_halted = running_probability < (1 - self.halting_thresh)

            #update_weights = still_running * running_probability
            new_target = self.decoder_layer(new_target, memory, tgt_mask=tgt_mask)

            target = new_target + target

            # update counter
            #self.dec_ponder_time[~new_halted] += 1

        return target

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a square matrix where each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

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

    #print(f"{len(batches)} batches of size {batch_size}")

    return batches

task_instance = MediumVaryingCountingTask()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UniversalTransformer(num_tokens=vocab_size, dim_model=EMB_DIM, num_heads=num_heads,
                             encoding_max_timestep=1, decoding_max_timestep=10, dropout_p=0.).to(device)
#opt = torch.optim.SGD(model.parameters(), lr=0.00001)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
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
        pred = torch.reshape(pred, [-1, vocab_size])
        y_expected = torch.reshape(y_expected, [-1])
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
                torch.save(model.state_dict(), 'UTFullCounter-transformer.pt')

    return train_loss_list

if RESUME_MODEL:
    # load model
    model.load_state_dict(torch.load('UTFullCounter-transformer.pt'))
    model = model.to(device)

    model.train()

if TRAIN_MODEL:
    train_loss_list = fit(model, opt, loss_fn, NUM_EPOCHS)
else:
    # load model
    model.load_state_dict(torch.load('UTFullCounter-transformer.pt'))
    model = model.double().to(device)

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
train_data = generate_random_data(100, task_instance, grid_dim=15)
train_dataloader = batchify_data(train_data, batch_size=100)

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

# # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
# # Get mask to mask out the next words
# sequence_length = y_input.size(1)
# tgt_mask = model.get_tgt_mask(sequence_length).to(device).double()
#
# # Standard training except we pass in y_input and tgt_mask
# pred = model(X, y_input, tgt_mask)
#
# # Permute pred to have batch size first again
# loss = loss_fn(pred, y_expected)
#
# # pred shape = [batch_size, vocab_size, seq_length=19 rather than 18!]
# # y_expected shape = [batch_size, 19 instead of 18!]
# print("pred shape = ", pred.shape)
# print("y expected shape = ", y_expected.shape)
#
# print("pred = ", pred.cpu().data.numpy())
# print("y expected = ", y_expected.cpu().data.numpy())
#
# print("loss = ", loss.cpu().data.numpy())
