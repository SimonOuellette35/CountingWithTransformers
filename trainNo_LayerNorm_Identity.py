import torch
import torch.nn as nn
import numpy as np
from models.FFNoLayerNorm import FFNoLayerNorm
import torch.optim as optim

# This is the training script for experiment No-LayerNorm-Identity.
# Set TRAIN_MODEL to True to train the model, and False to evaluate a pre-trained model.
# RESUME_MODEL can be set to True to resume training from a previous training session.

np.set_printoptions(suppress=True)

RESUME_MODEL = False
TRAIN_MODEL = True

train_batch_size = 200
source_vocab_size = 1  # dimensionality of each source token
hidden_dim = 10
num_epochs = 100000
test_batch_size = 1000
device = 'cuda'
LR = 0.0001
num_heads = 1
d_feedforward = 10
EMB_DIM = 5

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model = FFNoLayerNorm(EMB_DIM, batch_first=True).to(device).double()

if RESUME_MODEL:
    model.load_state_dict(torch.load('Identity-model-no-layernorm.pt'))
    model = model.double().to(device)
    model.train()
else:
    model.apply(init_weights)

criterion = nn.MSELoss()

def generate_data(is_training=True):
    X = []
    if is_training:
        for b_idx in range(train_batch_size):
            X.append(np.random.uniform(-0.5, 0.5, EMB_DIM))
    else:
        for b_idx in range(train_batch_size):
            X.append(np.random.uniform(-1., 1., EMB_DIM))

    return torch.from_numpy(np.array(X)).to(device), \
           torch.from_numpy(np.array(X)).to(device)

if TRAIN_MODEL:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    CLIP = 1.

    best_loss = np.inf

    train_losses = []
    for epoch in range(num_epochs):

        source, target = generate_data()

        preds = model(source.double())
        loss = criterion(preds.double(), target.to(device).double())

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        optimizer.step()

        with torch.no_grad():
            test_source, test_target = generate_data(is_training=False)

            test_preds = model(test_source.double())
            test_loss = criterion(test_preds.double(), test_target.double())

        print("Epoch: %i, loss = %.10f (test loss = %.6f)" % (epoch, loss, test_loss))

        train_losses.append(loss.cpu().data.numpy())
        if len(train_losses) >= 25:
            mean_loss = np.mean(train_losses[-25:])
            if mean_loss < best_loss:
                best_loss = mean_loss
                print("==> Saving new best model!")
                torch.save(model.state_dict(), 'Identity-model-no-layernorm.pt')

else:
    model.load_state_dict(torch.load('Identity-model-no-layernorm.pt'))
    model = model.double().to(device)
