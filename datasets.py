import torch
import numpy as np
import os
from midi_processor import processor
import utils


class MidiDataset:

    def __init__(self, files_dir, seq_length, subset_prop):
        self.files_dir = files_dir
        self.midi_paths = []
        self.seq_length = seq_length

        self.subset_prop = subset_prop

        self.data_x = []
        self.data_y = []
        for i in utils.progress_iter(range(len(os.listdir(self.files_dir))), "Preparing Data"):
            file = os.listdir(self.files_dir)[i]
            filename = os.fsdecode(file)
            if filename.endswith(".mid"):
                self.midi_paths.append(file)
            encoded_midi = processor.encode_midi(os.path.join(self.files_dir, file))
            for note_i in range(len(encoded_midi) - self.seq_length):
                seq_in = encoded_midi[note_i:note_i + self.seq_length]
                seq_out = encoded_midi[note_i + self.seq_length]
                self.data_x.append([note for note in seq_in])
                self.data_y.append(seq_out)

        self.data_x = torch.tensor(self.data_x, dtype=torch.float32).reshape(len(self.data_x), self.seq_length, 1)
        self.data_y = torch.tensor(self.data_y)
        self.vocab = torch.max(torch.max(self.data_y), torch.max(self.data_x)).to(dtype=torch.int64) + 1
        self.data_x /= self.vocab


    def __len__(self):
        return int(len(self.data_x)*self.subset_prop)

    def __getitem__(self, idx):
        return [self.data_x[idx], self.data_y[idx]]
        #return [torch.tensor(self.data_x[idx], dtype=torch.float32).reshape(len(self), self.seq_length, 1),
                #torch.tensor(self.data_y[idx], dtype=torch.float32)]

    def shuffle(self):
        shuffle_indices = torch.randperm(len(self.data_x))
        self.data_x = self.data_x[shuffle_indices]
        self.data_y = self.data_y[shuffle_indices]