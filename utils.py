import tqdm
import sys
import os


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def progress_iter(it, desc):
    return tqdm.tqdm(range(len(it)),
                desc=f'\t{desc}',
                unit=" batches",
                file=sys.stdout,
                colour="GREEN",
                bar_format="{desc}: {percentage:0.2f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}]")
