import torch
import random
import math
import numpy as np
import os
import processor
import utils


class MidiDatasetByPiece:
    def __init__(self, source_dir: str, chunks_dir: str, seq_length: int, batch_size: int, subset_prop: float, sample_size: int, save_chunks: bool, shuffle: bool):
        self.shuffle = shuffle
        self.loader = None
        self.source_dir = source_dir
        self.chunks_dir = chunks_dir
        self.midi_paths = []
        self.seq_length = seq_length
        self.sample_size = sample_size
        self.subset_prop = subset_prop
        self.batch_size = batch_size

        self.selected_pairs = []
        self.num_pieces = 0
        piece_idx = 0
        self.vocab = 0

        if save_chunks:
            for i in utils.progress_iter(range(len(os.listdir(self.source_dir))), "Saving Chunks"):
                file = os.listdir(self.source_dir)[i]
                filename = os.fsdecode(file)
                if filename.endswith(".mid") or filename.endswith(".midi"):

                    self.midi_paths.append(file)
                    encoded_midi = processor.encode_midi(os.path.join(self.source_dir, file))

                    if self.vocab < max(encoded_midi):
                        self.vocab = max(encoded_midi)

                    torch.save(encoded_midi, f"{self.chunks_dir}/{piece_idx}.midi.pth")
                    piece_idx += 1

            self.num_pieces = piece_idx
            torch.save([self.num_pieces, self.vocab+1], f"{self.chunks_dir}/info.pth")
            exit(0)
        else:
            self.num_pieces, self.vocab = torch.load(f"{self.chunks_dir}/info.pth")

    def __len__(self):
        return len(self.selected_pairs)

    def __getitem__(self, idx):
        return self.selected_pairs[idx]

    def create_loaders(self):
        # Select pieces based on sample_size
        pieces_indices = list(range(self.num_pieces))
        random.shuffle(pieces_indices)
        pieces_indices = pieces_indices[:self.sample_size]
        self.selected_pairs = self.load_saves_by_idx(pieces_indices)
        
        # Sample a subset of subset_prop
        pair_indices = list(range(len(self.selected_pairs)))
        random.shuffle(pair_indices)

        sampler = torch.utils.data.SubsetRandomSampler(
            pair_indices[:int(len(pair_indices) * self.subset_prop)])
        self.loader = torch.utils.data.DataLoader(self, batch_size=self.batch_size, sampler=sampler)

    def load_saves_by_idx(self, idx: list):
        data_x = []
        data_y = []
        data = []
        for i in idx:
            loaded_piece = torch.load(f"{self.chunks_dir}/{i}.midi.pth")
            for note_i in range(len(loaded_piece) - self.seq_length):
                piece_x = loaded_piece[note_i:note_i + self.seq_length]
                piece_y = loaded_piece[note_i + self.seq_length]
                data.append([piece_x, piece_y])
                data_x.append(piece_x)
                data_y.append(piece_y)

        data_x = torch.tensor(data_x, dtype=torch.float32).reshape(len(data_x), self.seq_length, 1)
        data_y = torch.tensor(data_y)
        data = [[data_x[i].int().squeeze(-1), data_y[i]] for i in range(len(data_y))]
        return data

    def print_info(self):
        print("-- Dataset Info --")
        print(f"Number of Chunks: {self.num_pieces}")
        print(f"Chunk Size: {len(self)}")
        print(f"\tTrain: {self.train_pieces_size} pieces")
        print(f"\tValidation: {self.valid_pieces_size} pieces")
        print(f"Vocabulary: {self.vocab.item()}")

def list_to_pairs(seq_list, seq_length):
    data_x = []
    data_y = []
    for note_i in range(len(seq_list) - seq_length):
        seq_in = seq_list[note_i:note_i + seq_length]
        seq_out = seq_list[note_i + seq_length]

        # Add note to chunk
        data_x.append([note for note in seq_in])
        data_y.append(seq_out)