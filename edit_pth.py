import torch
import os
from collections import OrderedDict

while True:
    os.system("clear")
    path = input("Enter path to begin: ")
    pth = torch.load(path)
    while True:
        printed_list = []
        i = 0
        for item in pth:

            if isinstance(item, torch.Tensor):
                if item.dim() == 0:
                    printed_list.append(f"[{i}] Scaler({item.item()})")
                else:
                    printed_list.append(f"[{i}] Tensor{list(item.shape)}")
            elif isinstance(item, OrderedDict):
                printed_list.append(f"[{i}] OrderedDict")
            else:
                printed_list.append(f"[{i}] {item}")

            i += 1
        print("[", end='\n\t')
        print(',\n\t'.join(printed_list))
        print("]")

        input()