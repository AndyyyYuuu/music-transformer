import torch
import random
import os
import numpy as np
import wandb

from midi_processor import processor
import model
import datasets
import utils

NUM_EPOCHS = 64
TRAIN_SPLIT = 0.8
SEQ_LENGTH = 100
LAYERS = 3
HIDDEN_SIZE = 256
DROPOUT_CHANCE = 0.2
NUM_HEADS = 4
EMBED_SIZE = 64

DO_WANDB = True
LOAD_FROM_MIDI = False

MODEL_NAME = "maestro-2"
SAVE_PATH = f"models/{MODEL_NAME}.pth"


def checkpoint(data):
    torch.save(data, SAVE_PATH)


if DO_WANDB:
    wandb.init(
        project="music-transformer",
        config={
            "architecture": "Transformer",
            "dataset": "MAESTRO",
            "train_split": TRAIN_SPLIT,
            "sequence_length": SEQ_LENGTH,
            "epochs": NUM_EPOCHS,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "num_heads": NUM_HEADS,
            "embed_size": EMBED_SIZE,
            "save_name": MODEL_NAME,
            "dropout": DROPOUT_CHANCE
        }
    )
print("Loading dataset...")

train_set = datasets.MidiDatasetByPiece(
    source_dir="data/maestro/midi_train",
    chunks_dir="data/maestro/tensor_train",
    seq_length=SEQ_LENGTH,
    subset_prop=0.1,
    sample_size=20,
    save_chunks=False,
    shuffle=True
)

valid_set = datasets.MidiDatasetByPiece(
    source_dir="data/maestro/midi_valid",
    chunks_dir="data/maestro/tensor_valid",
    seq_length=SEQ_LENGTH,
    subset_prop=0.1,
    sample_size=4,
    save_chunks=False,
    shuffle=False
)

train_set.vocab = max(train_set.vocab, valid_set.vocab)
valid_set.vocab = train_set.vocab

# dataset.print_info()

#composer = model.Composer(max(train_set.vocab, valid_set.vocab), LAYERS, HIDDEN_SIZE, DROPOUT_CHANCE)

composer = model.Composer(
    num_notes=max(train_set.vocab, valid_set.vocab),
    emb_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    hidden_size=HIDDEN_SIZE,
    num_layers=LAYERS,
    dropout_chance=DROPOUT_CHANCE

)

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
        composer = model.Composer(train_set.vocab, loaded_layers, loaded_hidden_size, loaded_dropout)
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


valid_set.create_loaders()
for epoch in range(start_epoch, NUM_EPOCHS):
    train_set.create_loaders()

    composer.train()
    loading_iter = iter(train_set.loader)
    for i in utils.progress_iter(range(len(train_set.loader)), "Training"):

        x_batch, y_batch = next(loading_iter)
        y_pred = composer(x_batch)

        # y_pred_flat = y_pred.view(-1, 388)
        loss = loss_function(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if DO_WANDB: wandb.log({"train_loss_mean": loss})



    composer.eval()
    loss = 0
    with torch.no_grad():
        loading_iter = iter(valid_set.loader)
        for i in utils.progress_iter(valid_set.loader, "Validating"):
            X_batch, y_batch = next(loading_iter)
            # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = composer(X_batch)

            loss += loss_function(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = composer.state_dict()
        print(f"Loss: {loss}")
        if DO_WANDB: wandb.log({"valid_loss_mean": loss})
        checkpoint([best_model, train_set.vocab, best_loss, epoch, composer.num_layers, composer.hidden_size, composer.dropout_chance])

