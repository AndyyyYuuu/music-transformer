# A Music Transformer
Training and running scripts for a PyTorch transformer that generates music. (Excuse the low-effort commit messages. In my defense, I was getting a little lost in the debugging to care, and now's too late to change them.)

## Methods

### Architecture
The architecture was the fairly standard Transformer decoder as seen in [Vaswani et al.](https://arxiv.org/abs/1706.03762). Throughout my experiments, its hyperparameters were typically as follows: 
- 256 - 2048 MIDI events of context length
- ~3 Transformer encoder layers
- 128 - 512 hidden linear layers
- 2-8 attention heads
- 32- to 126-dimensioned embedding vectors

### Training Data
Training data was a combination of MIDI piano pieces (primarily classical and romantic) from the [MAESTRO Dataset](https://arxiv.org/abs/1810.12247) and [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano), resulting in around 10,000 pieces. 

### Data Processing

The [midi-neural-processor](https://github.com/jason9693/midi-neural-processor/tree/3875298892adafbc5cebeeb32050732c5ae5aee1) submodule was used. MIDI data was processed into 1-D tensors of integers from 0-387, each representing a MIDI event (change in note on/off status, time shift, and change in note dynamics), then saved as .pt files in a folder. 

The train-test split was 80%-20%.

### Training
Adam was used for optimization. Before each epoch, a random mini-batch of .pt pieces was loaded, from which random sequence-prediction pairs were sampled for training. 
I used [Weights & Biases](https://wandb.ai/site) to track my training progress. 

### Compute
I rented compute from [Vast.ai](https://vast.ai/) to train my project. Training scripts were pulled from the Github onto the virtual machines using a few [.ipynb commands](https://github.com/AndyyyYuuu/music-transformer/blob/main/train.ipynb). 

## Results
Test loss achieved 3.99, well below the expected value. Unfortunately, the music generated was still audibly chaos. 

https://github.com/user-attachments/assets/ad342645-8d2c-458d-bb92-6caeb737348e

It was good fun though. 
