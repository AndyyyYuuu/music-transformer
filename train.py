import torch
import random
import os
import numpy as np
import wandb

from midi_processor import processor
import model
import datasets
import utils

NUM_EPOCHS = 40
TRAIN_SPLIT = 0.8
SEQ_LENGTH = 100
LAYERS = 2
HIDDEN_SIZE = 128
DROPOUT_CHANCE = 0.2

DO_WANDB = True
LOAD_FROM_MIDI = True

MODEL_NAME = "maestro-2"
SAVE_PATH = f"models/{MODEL_NAME}.pth"


def checkpoint(data):
    torch.save(data, SAVE_PATH)


if DO_WANDB:
    wandb.init(
        project="music-lstm",
        config={
            "architecture": "LSTM",
            "dataset": "MAESTRO",
            "train_split": TRAIN_SPLIT,
            "sequence_length": SEQ_LENGTH,
            "epochs": NUM_EPOCHS,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "save_name": MODEL_NAME,
            "dropout": DROPOUT_CHANCE
        }
    )
print("Loading dataset...")
'''
dataset = datasets.ChunkedMidiDataset(
    source_dir="data/maestro",
    chunks_dir="data/maestro/tensor",
    seq_length=SEQ_LENGTH,
    train_split=TRAIN_SPLIT,
    subset_prop=0.1,
    chunk_size=100000,
    save_chunks=True
)
'''
dataset = datasets.MidiDatasetByPiece(
    source_dir="data/maestro/midi",
    chunks_dir="data/maestro/tensor",
    seq_length=SEQ_LENGTH,
    train_split=TRAIN_SPLIT,
    subset_prop=0.1,
    sample_size=50,
    save_chunks=False
)

# dataset.print_info()

composer = model.Composer(dataset.vocab, LAYERS, HIDDEN_SIZE, DROPOUT_CHANCE)

optimizer = torch.optim.Adam(composer.parameters())
loss_function = torch.nn.CrossEntropyLoss(reduction="mean")

best_model = None
best_loss = np.inf

start_epoch = 0

# Load checkpoint
if os.path.exists(SAVE_PATH):
    past_state_dict = torch.load(SAVE_PATH)
    loaded_best_model, loaded_vocab, loaded_best_loss, loaded_epoch, loaded_layers, loaded_hidden_size, loaded_dropout = past_state_dict
    if loaded_epoch < NUM_EPOCHS-1:
        best_model = loaded_best_model
        best_loss = loaded_best_loss
        start_epoch = loaded_epoch+1
        composer = model.Composer(dataset.vocab, loaded_layers, loaded_hidden_size, loaded_dropout)
        composer.load_state_dict(best_model)
        print("LOADED MODEL")
        print(f"Epochs to train: {NUM_EPOCHS-start_epoch}")
        print(f"Save path: {SAVE_PATH}")
        print(f"Layers: {loaded_layers}")
        print(f"Hidden size: {loaded_hidden_size}")
        input("Enter to resume training >>> ")
    else:
        print(f"Hey Andy, this model is already trained up to {NUM_EPOCHS} epochs.")
        exit(0)
else:
    print("TRAIN NEW MODEL")
    print(f"Epochs to train: {NUM_EPOCHS}")
    print(f"Save path: {SAVE_PATH}")



for epoch in range(start_epoch, NUM_EPOCHS):

    dataset.randomize_loaders()

    composer.train()
    loading_iter = iter(dataset.train_loader)
    for i in utils.progress_iter(range(len(dataset.train_loader)), "Training"):

        x_batch, y_batch = next(loading_iter)
        y_pred = composer(x_batch)
        loss = loss_function(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if DO_WANDB: wandb.log({"train_loss_mean": loss})



    composer.eval()
    loss = 0
    with torch.no_grad():
        loading_iter = iter(dataset.valid_loader)
        for i in utils.progress_iter(dataset.valid_loader, "Validating"):
            X_batch, y_batch = next(loading_iter)
            # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = composer(X_batch)
            loss += loss_function(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = composer.state_dict()
        print(f"Loss: {loss}")
        if DO_WANDB: wandb.log({"valid_loss_mean": loss})
        checkpoint([best_model, dataset.vocab, best_loss, epoch, composer.layers, composer.hidden_size, composer.dropout_chance])

