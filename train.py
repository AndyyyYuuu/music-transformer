import torch
import numpy as np
import wandb


from midi_processor import processor
import model
import datasets

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

composer = model.Composer(32)

optimizer = torch.optim.Adam(composer.parameters())
loss_function = torch.nn.CrossEntropyLoss(reduction="sum")

for epoch in range(NUM_EPOCHS):
    composer.train()
    loading_iter = iter(train_loader)
    for i in train_loader:

        x_batch, y_batch = next(loading_iter)
        print(x_batch.size())
        print(composer(x_batch))
        exit()

