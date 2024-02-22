import torch
import numpy as np
import wandb


from midi_processor import processor
import model
import datasets
import utils

NUM_EPOCHS = 10
TRAIN_SPLIT = 0.8
SEQ_LENGTH = 100


print("Loading dataset...")
dataset = datasets.MidiDataset("dataset", SEQ_LENGTH)

train_size = int(len(dataset)*TRAIN_SPLIT)
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

print("-- Dataset Info --")
print(f"Size: {len(dataset)}")
print(f"\tTrain: {train_size}")
print(f"\tTest: {test_size}")
print(dataset.vocab)
composer = model.Composer(dataset.vocab)

optimizer = torch.optim.Adam(composer.parameters())
loss_function = torch.nn.CrossEntropyLoss(reduction="sum")

for epoch in range(NUM_EPOCHS):
    composer.train()
    loading_iter = iter(train_loader)
    for i in utils.progress_iter(range(len(train_loader)), "Training"):

        x_batch, y_batch = next(loading_iter)
        y_pred = composer(x_batch)
        loss = loss_function(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(y_pred)

