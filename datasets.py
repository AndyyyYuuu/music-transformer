import torch
import random
import math
import numpy as np
import os
from midi_processor import processor
import utils


class MidiDataset:

    def __init__(self, files_dir, seq_length, train_split, subset_prop):

        self.train_loader = None
        self.valid_loader = None
        self.files_dir = files_dir
        self.midi_paths = []
        self.seq_length = seq_length

        self.subset_prop = subset_prop

        self.data_x = []
        self.data_y = []
        for i in utils.progress_iter(range(len(os.listdir(self.files_dir))), "Preparing Data"):
            file = os.listdir(self.files_dir)[i]
            filename = os.fsdecode(file)
            if filename.endswith(".mid") or filename.endswith(".midi"):
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

        self.train_size = int(len(self.data_x) * train_split)
        self.valid_size = len(self.data_x) - self.train_size

        self.train_indices = list(range(len(self.data_x)))[:self.train_size]
        self.valid_indices = list(range(len(self.data_x)))[self.train_size:]

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return [self.data_x[idx], self.data_y[idx]]
        #return [torch.tensor(self.data_x[idx], dtype=torch.float32).reshape(len(self), self.seq_length, 1),
                #torch.tensor(self.data_y[idx], dtype=torch.float32)]

    def randomize_loaders(self):
        random.shuffle(self.train_indices)
        random.shuffle(self.valid_indices)
        train_sampler = torch.utils.data.SubsetRandomSampler(self.train_indices[:int(len(self.train_indices) * self.subset_prop)])
        valid_sampler = torch.utils.data.SubsetRandomSampler(self.valid_indices[:int(len(self.valid_indices) * self.subset_prop)])
        self.train_loader = torch.utils.data.DataLoader(self, batch_size=32, sampler=train_sampler)
        self.valid_loader = torch.utils.data.DataLoader(self, batch_size=32, sampler=valid_sampler)

    def print_info(self):
        print("-- Dataset Info --")
        print(f"Size: {len(self)}")
        print(f"\tTrain: {self.train_size}")
        print(f"\tValidation: {self.valid_size}")
        print(f"Vocabulary:{self.vocab.item()}")


# Dataset with input-output pairs saved as chunks
class ChunkedMidiDataset:

    def __init__(self, source_dir, chunks_dir, seq_length, train_split, subset_prop, chunk_size, save_chunks):

        self.train_loader = None
        self.valid_loader = None
        self.source_dir = source_dir
        self.chunks_dir = chunks_dir
        self.midi_paths = []
        self.seq_length = seq_length
        self.chunk_size = chunk_size
        self.subset_prop = subset_prop

        self.data_x = []
        self.data_y = []
        self.chunk_x = []
        self.chunk_y = []
        self.num_chunks = 0
        chunk_x = []
        chunk_y = []
        chunk_idx = 0
        self.vocab = 0
        if save_chunks:
            for i in utils.progress_iter(range(len(os.listdir(self.source_dir))), "Saving Chunks"):
                file = os.listdir(self.source_dir)[i]
                filename = os.fsdecode(file)
                if filename.endswith(".mid") or filename.endswith(".midi"):
                    self.midi_paths.append(file)
                    encoded_midi = processor.encode_midi(os.path.join(self.source_dir, file))
                    for note_i in range(len(encoded_midi) - self.seq_length):
                        seq_in = encoded_midi[note_i:note_i + self.seq_length]
                        seq_out = encoded_midi[note_i + self.seq_length]

                        # Find min and max
                        if seq_out > self.vocab:
                            self.vocab = seq_out

                        # Add note to chunk
                        chunk_x.append([note for note in seq_in])
                        chunk_y.append(seq_out)

                    if len(chunk_y) >= self.chunk_size:
                        self.save_chunk(chunk_x, chunk_y, chunk_idx)
                        chunk_idx += 1
            self.num_chunks = chunk_idx
            # Ignore data not included in chunk?
        else:
            # Count chunks
            for i in range(len(os.listdir(f"{self.chunks_dir}"))):
                file = os.listdir(self.chunks_dir)[i]
                filename = os.fsdecode(file)
                if filename.endswith(".chunk"):
                    self.num_chunks += 1
            # Find chunk vocab
            for chunk_i in range(self.num_chunks):
                chunk_x, chunk_y = self.load_chunk_n(chunk_i)
                for i in chunk_y:
                    if chunk_y[i] > self.vocab:
                        self.vocab = chunk_y[i]

        self.train_size = int(self.chunk_size * train_split)
        self.valid_size = self.chunk_size - self.train_size

        self.train_indices = list(range(self.chunk_size))[:self.train_size]
        self.valid_indices = list(range(self.chunk_size))[self.train_size:]

    def save_chunk(self, chunk_x, chunk_y, chunk_idx):
        chunk_x = torch.tensor(chunk_x, dtype=torch.float32).reshape(len(chunk_x), self.seq_length, 1)
        chunk_y = torch.tensor(chunk_y)
        torch.save((chunk_x, chunk_y), f"{self.chunks_dir}/{chunk_idx}.chunk")

    def load_chunk_n(self, n: int):
        return torch.load(f"{self.chunks_dir}/{n}.chunk")

    def __len__(self):
        return self.chunk_size

    def __getitem__(self, idx):
        return [self.chunk_x[idx], self.chunk_y[idx]]
        # return [torch.tensor(self.data_x[idx], dtype=torch.float32).reshape(len(self), self.seq_length, 1),
        # torch.tensor(self.data_y[idx], dtype=torch.float32)]

    def randomize_loaders(self):
        # Load a random chunk to sample from each epoch
        self.chunk_x, self.chunk_y = self.load_chunk_n(random.randint(self.num_chunks))
        random.shuffle(self.train_indices)
        random.shuffle(self.valid_indices)
        train_sampler = torch.utils.data.SubsetRandomSampler(
            self.train_indices[:int(len(self.train_indices) * self.subset_prop)])
        valid_sampler = torch.utils.data.SubsetRandomSampler(
            self.valid_indices[:int(len(self.valid_indices) * self.subset_prop)])
        self.train_loader = torch.utils.data.DataLoader(self, batch_size=32, sampler=train_sampler)
        self.valid_loader = torch.utils.data.DataLoader(self, batch_size=32, sampler=valid_sampler)

    def print_info(self):
        print("-- Dataset Info --")
        print(f"Total Pairs: {self.num_chunks*len(self)}")
        print(f"Number of Chunks: {self.num_chunks}")
        print(f"Chunk Size: {len(self)}")
        print(f"\tTrain: {self.train_size}")
        print(f"\tValidation: {self.valid_size}")
        print(f"Vocabulary:{self.vocab.item()}")


# Dataset
class MidiDatasetByPieceOld:
    def __init__(self, source_dir, chunks_dir, seq_length, train_split, subset_prop, sample_size, save_chunks):

        self.train_loader = None
        self.valid_loader = None
        self.source_dir = source_dir
        self.chunks_dir = chunks_dir
        self.midi_paths = []
        self.seq_length = seq_length
        self.sample_size = sample_size
        self.subset_prop = subset_prop

        self.selected_train = []
        self.selected_valid = []
        self.num_pieces = 0
        piece_idx = 0
        self.vocab = 0
        self.train_size = None
        self.valid_size = None
        self.train_indices = None
        self.valid_indices = None
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
        else:
            self.num_pieces, self.vocab = torch.load(f"{self.chunks_dir}/info.pth")

        self.train_pieces_size = int(self.sample_size * train_split)
        self.valid_pieces_size = self.sample_size - self.train_pieces_size

        self.train_pieces_indices = list(range(self.sample_size))[:self.train_pieces_size]
        self.valid_pieces_indices = list(range(self.sample_size))[self.train_pieces_size:]

    def __len__(self):
        return self.train_size + self.valid_size

    def __getitem__(self, idx):
        return (self.selected_train+self.selected_valid)[idx]

    def randomize_loaders(self):

        random.shuffle(self.train_pieces_indices)
        random.shuffle(self.valid_pieces_indices)
        self.selected_train = self.load_saves_by_idx(self.train_pieces_indices)

        self.train_size = len(self.selected_train)
        self.selected_valid = self.load_saves_by_idx(self.valid_pieces_indices)
        self.valid_size = len(self.selected_valid)
        print(self.valid_size)

        self.train_indices = range(self.train_size)
        self.valid_indices = range(self.train_size, self.train_size+self.valid_size)
        train_sampler = torch.utils.data.SubsetRandomSampler(
            self.train_indices[:int(len(self.train_indices) * self.subset_prop)])
        valid_sampler = torch.utils.data.SubsetRandomSampler(
            self.valid_indices[:int(len(self.valid_indices) * self.subset_prop)])
        self.train_loader = torch.utils.data.DataLoader(self, batch_size=32, sampler=train_sampler)
        self.valid_loader = torch.utils.data.DataLoader(self, batch_size=32, sampler=valid_sampler)


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

        data_x = torch.tensor(data_x, dtype=torch.float32).reshape(len(data_x), self.seq_length, 1) / float(self.vocab)
        data_y = torch.tensor(data_y)
        data = [[data_x[i], data_y[i]] for i in range(len(data_y))]
        return data

    def print_info(self):
        print("-- Dataset Info --")
        print(f"Number of Chunks: {self.num_pieces}")
        print(f"Chunk Size: {len(self)}")
        print(f"\tTrain: {self.train_pieces_size} pieces")
        print(f"\tValidation: {self.valid_pieces_size} pieces")
        print(f"Vocabulary: {self.vocab.item()}")


class MidiDatasetByPiece:
    def __init__(self, source_dir: str, chunks_dir: str, seq_length: int, subset_prop: float, sample_size: int, save_chunks:bool, shuffle:bool):
        self.shuffle = shuffle
        self.loader = None
        self.source_dir = source_dir
        self.chunks_dir = chunks_dir
        self.midi_paths = []
        self.seq_length = seq_length
        self.sample_size = sample_size
        self.subset_prop = subset_prop

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
        self.loader = torch.utils.data.DataLoader(self, batch_size=32, sampler=sampler)

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

        data_x = torch.tensor(data_x, dtype=torch.float32).reshape(len(data_x), self.seq_length, 1) / float(self.vocab)
        data_y = torch.tensor(data_y)
        data = [[data_x[i], data_y[i]] for i in range(len(data_y))]
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