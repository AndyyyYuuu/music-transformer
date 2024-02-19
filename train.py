import torch
import numpy as np
import wandb

from midi_processor import processor
import model
import datasets

TRAIN_SPLIT = 0.8

print("Loading dataset...")
dataset = datasets.MidiDataset("dataset")

train_size = int(len(dataset)*TRAIN_SPLIT)
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

print("-- Dataset Info --")
print(f"Size: {len(dataset)}")
print(f"\tTrain: {train_size}")
print(f"\tTest: {test_size}")

composer = model.Composer()