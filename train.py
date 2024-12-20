import torch
import random
import os
import numpy as np
import sys

import processor
import model
import datasets
import utils
import train_config
import importlib
importlib.reload(utils)
importlib.reload(train_config)
importlib.reload(datasets)
importlib.reload(model)

config = train_config.config

NUM_EPOCHS = config["num_epochs"]
MODEL_ID = config['model_id']
DATA_PATH = "data/"

utils.create_directory("models")

if len(sys.argv) > 1:
    try:
        MODEL_ID = int(sys.argv[1])
    except ValueError as e:
        pass

if len(sys.argv) > 2:
    DATA_PATH = sys.argv[2]

DO_WANDB = "wandb" in sys.argv
LOAD_FROM_MIDI = False

MODEL_NAME = f"piano-{MODEL_ID}"
SAVE_PATH = f"models/{MODEL_NAME}.pth"

if torch.cuda.is_available():
    print("CUDA device available.")
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    print("MPS device available.")
    device = torch.device("mps")

else:
    print("Devices not found. Using CPU.")
    device = torch.device("cpu")



def checkpoint(data):
    torch.save(data, SAVE_PATH)


try:
    import wandb
    if DO_WANDB:
        wandb.init(
            project="music-transformer",
            config={
                "dataset": "MAESTRO",
                **config["model"]
            }
        )
except ImportError:
    print("Warning: wandb is not installed")

print("Loading dataset...")

train_set = datasets.MidiDatasetByPiece(
    source_dir=os.path.join(DATA_PATH, "midi_train"),
    chunks_dir=os.path.join(DATA_PATH, "tensor_train"),
    batch_size=config["data"]["batch_size"],
    seq_length=config["model"]["sequence_length"],
    subset_prop=config["data"]["subset_prop"],
    sample_size=config["data"]["train_set_size"],
    save_chunks=False,
    shuffle=True,
    device=device
)

valid_set = datasets.MidiDatasetByPiece(
    source_dir=os.path.join(DATA_PATH, "midi_valid"),
    chunks_dir=os.path.join(DATA_PATH, "tensor_valid"),
    batch_size=config["data"]["batch_size"],
    seq_length=config["model"]["sequence_length"],
    subset_prop=config["data"]["subset_prop"],
    sample_size=config["data"]["valid_set_size"],
    save_chunks=False,
    shuffle=False,
    device=device
)

train_set.vocab = max(train_set.vocab, valid_set.vocab)
valid_set.vocab = train_set.vocab
config["training_info"]["vocab"] = train_set.vocab
config["model"]["vocab"] = train_set.vocab

# dataset.print_info()

#composer = model.Composer(max(train_set.vocab, valid_set.vocab), LAYERS, HIDDEN_SIZE, DROPOUT_CHANCE)

composer = model.Composer(config["model"])



best_model = None
best_loss = np.inf

start_epoch = 0

optimizer = torch.optim.Adam(composer.parameters(), lr=config["learning_rate"])
loss_function = torch.nn.CrossEntropyLoss(reduction="mean")

# Load checkpoint
if os.path.exists(SAVE_PATH):
    past_state_dict = torch.load(SAVE_PATH, map_location=torch.device("cpu"))
    best_model, config, optim_state = past_state_dict
    optimizer.load_state_dict(optim_state)
    train_info = config["training_info"]
    if train_info["epoch_at"] < NUM_EPOCHS-1:
        best_loss = train_info["min_loss"]
        start_epoch = train_info["epoch_at"]+1
        composer = model.Composer(config["model"])
        composer.load_state_dict(best_model)
        print("LOADED MODEL")
        print(f"Epochs to train: {NUM_EPOCHS-start_epoch}")
        print(f"Save path: {SAVE_PATH}")
        print(f"Layers: {config['model']['layers']}")
        print(f"Hidden size: {config['model']['hidden_size']}")
        input("Enter to resume training >>> ")
    else:
        print(f"Hey Andy, this model is already trained up to {NUM_EPOCHS} epochs.")
        exit(0)
else:
    print("TRAIN NEW MODEL")
    print(f"Epochs to train: {NUM_EPOCHS}")
    print(f"Save path: {SAVE_PATH}")



composer.to(device)
valid_set.create_loaders()

for epoch in range(start_epoch, NUM_EPOCHS):

    train_set.create_loaders()

    composer.train()
    loading_iter = iter(train_set.loader)
    total_loss = 0
    for i in utils.progress_iter(range(len(train_set.loader)), "Training"):

        x_batch, y_batch = next(loading_iter)
        # x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = composer(x_batch)

        # y_pred_flat = y_pred.view(-1, 388)
        loss = loss_function(y_pred, y_batch)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if DO_WANDB: wandb.log({"train_loss_mean": loss})
    print(f"Train Loss: {total_loss/i}")



    composer.eval()
    loss = 0
    with torch.no_grad():
        loading_iter = iter(valid_set.loader)
        for i in utils.progress_iter(valid_set.loader, "Validating"):
            x_batch, y_batch = next(loading_iter)
            # x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = composer(x_batch)

            loss += loss_function(y_pred, y_batch)
        loss /= len(valid_set.loader)
        if loss < best_loss:
            best_loss = loss
            best_model = composer.state_dict()
        print(f"Valid Loss: {loss}")
        if DO_WANDB: wandb.log({"valid_loss_mean": loss})

        config["training_info"]["epoch_at"] = epoch
        config["training_info"]["min_loss"] = best_loss
        # checkpoint([best_model, train_set.vocab, best_loss, epoch, SEQ_LENGTH, composer.num_layers, composer.hidden_size, composer.dropout_chance, composer.emb_size, composer.num_heads])
        checkpoint([best_model, config, optimizer.state_dict()])

