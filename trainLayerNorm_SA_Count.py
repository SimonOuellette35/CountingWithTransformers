import torch
import torch.nn as nn
import numpy as np
import tasks.counting_tasks as tasks
import utils.transformer_utils as utils
from models.LayerNorm_SA_Count import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# Inputs: (flattened) grids with randomized pixel colors.
# Outputs: [<color token>, <count value>, <color token>, <count value>, etc. (for all non-zeros pixel colors other than background color)]

np.set_printoptions(suppress=True)

RESUME_MODEL = False
TRAIN_MODEL = False

GRID_DIM = 7

#source_vocab_size = 1  # dimensionality of each source token
#hidden_dim = 10
num_epochs = 200000
test_batch_size = 1000
device = 'cuda'
LR = 0.0002
num_heads = 1
train_batch_size = 50

EMB_DIM = 10
def one_hot_encode(x):
    output = torch.zeros((x.shape[0], x.shape[1], 10)).to('cuda')

    for b_idx in range(x.shape[0]):
        for step_idx in range(x.shape[1]):
            tmp = torch.zeros(10)
            tmp[int(x[b_idx, step_idx].cpu().data.numpy())] = 1.

            output[b_idx, step_idx, :] = tmp

    return output

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

# use the sample generator to generate training and test samples
current_task = {
    0: [tasks.BasicCountingV3],
    1: [],
    2: []
}

def preprocessTarget(source, target):
    # for each token in the source sequence, fetch the corresponding target count and place it at the corresponding
    # color index in a 10-dim array (the rest being zeros). If the color is 0, the 10-dim array is all zeros.
    # Predictions will have shape: [batch_size, sequence_length, 10]
    # Target will have shape: [batch_size, sequence_length, 10]

    new_target = torch.zeros_like(source).to(device)
    for b_idx in range(source.shape[0]):
        for step_idx in range(source.shape[1]):
            color_idx = int(torch.argmax(source[b_idx, step_idx]).cpu().data.numpy())
            target_vec = torch.zeros(10)
            if color_idx > 0:
                target_vec[color_idx] = target[b_idx, color_idx-1]

            new_target[b_idx, step_idx, :] = target_vec

    return new_target

# def preprocessTarget(source, target):
#     # for each token in the source sequence, fetch the corresponding target count and place it at the corresponding
#     # color index in a 10-dim array (the rest being zeros).
#     # Predictions will have shape: [batch_size, sequence_length, 10]
#     # Target will have shape: [batch_size, sequence_length, 10]
#
#     new_target = torch.zeros((source.shape[0], source.shape[1], EMB_DIM)).to(device)
#     for b_idx in range(source.shape[0]):
#         for step_idx in range(source.shape[1]):
#             color_idx = int(torch.argmax(source[b_idx, step_idx]).cpu().data.numpy())
#             target_vec = torch.zeros(10)
#             target_vec[color_idx] = target[b_idx, color_idx]
#
#             new_target[b_idx, step_idx, :10] = target_vec
#
#     return new_target

enc_layer = TransformerEncoderLayer(d_model=EMB_DIM, nhead=num_heads, batch_first=True).to(device).double()
model = TransformerEncoder(enc_layer, num_layers=1).to(device).double()

if RESUME_MODEL:
    model.load_state_dict(torch.load('LayerNorm-SA-Count.pt'))
    model = model.double().to(device)
    model.train()
else:
    model.apply(init_weights)

criterion = nn.MSELoss()

# show training vs. validation loss (MSE) and accuracy
def single_pred_accuracy(pred, tgt):
    acc = 0.

    for cell_idx in range(pred.shape[0]):
        current_preds = np.round(pred[cell_idx].cpu().data.numpy())
        if np.all(current_preds == tgt[cell_idx].cpu().data.numpy()):
            acc += 1.

    return acc / float(pred.shape[0])

def batch_accuracy(pred, tgt):
    acc = 0
    for batch_idx in range(pred.shape[0]):
        batch_acc = 0.

        for cell_idx in range(pred.shape[1]):
            current_preds = np.round(pred[batch_idx, cell_idx].cpu().data.numpy())
            if np.all(current_preds == tgt[batch_idx, cell_idx].cpu().data.numpy()):
                batch_acc += 1.

        acc += batch_acc / float(pred.shape[1])

    return acc / float(pred.shape[0])

if TRAIN_MODEL:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    CLIP = 1.

    best_loss = np.inf

    train_losses = []
    for epoch in range(num_epochs):

        grid_dim = np.random.choice(np.arange(1, GRID_DIM))

        task_instance = current_task[0][0](grid_dim_min=grid_dim, grid_dim_max=grid_dim, num_px_max=1000)
        data_generator = utils.UTTaskDataGenerator(task_instance, input_grid_dim=grid_dim, output_grid_dim=3)

        length = grid_dim * grid_dim
        source, target, _, _ = data_generator.get_batch(length, train_batch_size)

        one_hot_source = one_hot_encode(source)

        target = preprocessTarget(one_hot_source, target)

        preds = model(one_hot_source.double())

        loss = criterion(preds.double(), target.to(device).double())

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        optimizer.step()

        acc = batch_accuracy(preds, target)
        print("Epoch: %i, loss = %.6f (accuracy = %.2f)" % (epoch, loss, acc))

        train_losses.append(loss.cpu().data.numpy())
        if len(train_losses) >= 25:
            mean_loss = np.mean(train_losses[-25:])
            if mean_loss < best_loss:
                best_loss = mean_loss
                print("==> Saving new best model!")
                torch.save(model.state_dict(), 'LayerNorm-SA-Count.pt')

else:
    model.load_state_dict(torch.load('LayerNorm-SA-Count.pt'))
    model = model.double().to(device)

model.eval()

print("Evaluating...")

TEST_GRID_DIM = 6
task_instance = current_task[0][0](grid_dim_min=TEST_GRID_DIM, grid_dim_max=TEST_GRID_DIM, num_px_max=1000)
data_generator = utils.UTTaskDataGenerator(task_instance, input_grid_dim=TEST_GRID_DIM, output_grid_dim=3)

length = TEST_GRID_DIM * TEST_GRID_DIM

accuracies = []
source, target, _, _ = data_generator.get_batch(length, 1000)

# print("source = ", source[0].cpu().data.numpy())
# print("target = ", target[0].cpu().data.numpy())

with torch.no_grad():
    one_hot_source = one_hot_encode(source)
    target = preprocessTarget(one_hot_source, target)

    count_targets = []
    for b_idx in range(target.shape[0]):
        for cell_target in target[b_idx].cpu().data.numpy():
            count_val = np.max(cell_target)
            count_targets.append(count_val)

    def getCountPerColor(vector):
        counts = np.zeros(10)
        for v in vector:
            counts[v] += 1

        return counts

    count_preds = []
    success_std_preds = []
    failure_std_preds = []
    for b_idx in range(source.shape[0]):
        source_elem = torch.reshape(one_hot_source[b_idx].double(), [1, source.shape[1], EMB_DIM])
        print("=================================================================================================")
        print("source = ", source[b_idx].cpu().data.numpy())
        print("count per color = ", getCountPerColor(source[b_idx].cpu().data.numpy()))
        #preds, x1, x2 = model(source_elem)
        preds = model(source_elem)

        num_preds = np.round(np.max(preds[0].cpu().data.numpy(), axis=-1))
        print("Count predictions = ", num_preds)

        num_tgts = np.round(np.max(target[b_idx].cpu().data.numpy(), axis=-1))
        print("Count targets = ", num_tgts)
        #print("Predictions = ", preds[0].cpu().data.numpy())

        acc = single_pred_accuracy(preds[0], target[b_idx])
        print("Accuracy = ", acc)
        accuracies.append(acc)

        # for cell_idx, s in enumerate(x1[0]):
        #     current_preds = np.round(preds[0, cell_idx].cpu().data.numpy())
        #     if np.all(current_preds == target[b_idx, cell_idx].cpu().data.numpy()):
        #         success_std_preds.append(np.std(s))
        #     else:
        #         failure_std_preds.append(np.std(s))

        for cell_pred in preds[0].cpu().data.numpy():
            count_val = np.max(cell_pred)
            count_preds.append(count_val)

    sns.set(font_scale=1.8)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_xlim(0, 10)
    ax1.set_xticks(np.arange(10))
    ax2.set_xlim(0, 10)
    ax2.set_xticks(np.arange(10))
    ax1.set_ylim(0, 0.21)
    ax2.set_ylim(0, 0.21)

    sns.histplot(ax=ax1, data=count_preds, binwidth=1, stat="density")
    ax1.set_title("Distribution of predicted count values")

    sns.histplot(ax=ax2, data=count_targets, binwidth=1, stat="density")
    ax2.set_title("Distribution of ground truth count values")
    plt.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    #
    # ax1.set_xlim(0.0, 0.0025)
    # ax1.set_xticks([0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025])
    # ax1.set_ylim(0, 6000)
    # ax2.set_ylim(0, 6000)
    # ax2.set_xlim(0.0, 0.0025)
    # ax2.set_xticks([0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025])
    #
    # sns.histplot(ax=ax1, data=success_std_preds, stat="density")
    # ax1.set_title("Distribution of stdev for successful preds")
    #
    # sns.histplot(ax=ax2, data=failure_std_preds, stat="density")
    # ax2.set_title("Distribution of stdev for failure cases")
    # plt.show()

print("Mean accuracy = ", np.mean(accuracies))