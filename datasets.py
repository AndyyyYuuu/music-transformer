import torch
import numpy as np
import os
from midi_processor import processor


class MidiDataset:

    def __init__(self, files_dir):
        self.files_dir = files_dir
        self.midi_paths = []
        for file in os.listdir("dataset"):
            filename = os.fsdecode(file)
            if filename.endswith(".mid"):
                self.midi_paths.append(file)

    def __len__(self):
        return len(self.midi_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.files_dir, self.midi_paths[idx])
        return processor.encode_midi(path)

