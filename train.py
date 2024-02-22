import torch
import os
import numpy as np
import wandb

from midi_processor import processor
import model
import datasets
import utils

NUM_EPOCHS = 5
TRAIN_SPLIT = 0.8
SEQ_LENGTH = 100

SAVE_PATH = "models/jazz-1.pth"

def checkpoint(best_model, best_loss, epoch):
    torch.save([best_model, best_loss, epoch], SAVE_PATH)



print("Loading dataset...")
dataset = datasets.MidiDataset("dataset", SEQ_LENGTH, subset_prop=0.1)

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

best_model = None
best_loss = np.inf

# Load checkpoint
if os.path.exists(SAVE_PATH):
    past_state_dict = torch.load(SAVE_PATH)
    loaded_best_model, loaded_best_loss, loaded_epoch = past_state_dict
    if loaded_epoch < NUM_EPOCHS-1:
        best_model = loaded_best_model
        best_loss = loaded_best_loss
        start_epoch = loaded_epoch+1
        print("LOADED MODEL")
        print(f"Epochs to train: {NUM_EPOCHS-start_epoch}")
        print(f"Save path: {SAVE_PATH}")
        input("Enter to resume training >>> ")
    else:
        print(f"Hey Andy, this model is already trained up to {NUM_EPOCHS} epochs.")
        exit(0)
else:
    print("TRAIN NEW MODEL")
    print(f"Epochs to train: {NUM_EPOCHS}")
    print(f"Save path: {SAVE_PATH}")



for epoch in range(NUM_EPOCHS):
    dataset.shuffle()
    composer.train()
    loading_iter = iter(train_loader)
    for i in utils.progress_iter(range(len(train_loader)), "Training"):

        x_batch, y_batch = next(loading_iter)
        y_pred = composer(x_batch)
        loss = loss_function(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    composer.eval()
    loss = 0
    with torch.no_grad():
        loading_iter = iter(test_loader)
        for i in utils.progress_iter(test_loader, "Validating"):
            X_batch, y_batch = next(loading_iter)
            # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = composer(X_batch)
            loss += loss_function(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = composer.state_dict()
        print(f"Loss: {loss}")
        checkpoint(best_model, best_loss, epoch)

