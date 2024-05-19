import torch
import numpy as np

import model
from midi_processor import processor

PATH = "models/maestro-1.pth"
SAVE_PATH = "results/maestro-1-2.mid"
PROMPTS_PATH = "data/maestro/midi_train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--3.midi"
# PROMPTS_PATH = "data/weimar/ArtPepper_Anthropology_FINAL.mid"

best_model, num_vocab, best_loss, epoch, layers, hidden_size, dropout = torch.load(PATH)

print(f"Epochs: {epoch}")
print(f"Layers: {layers}")
print(f"Hidden size: {hidden_size}")

TEMPERATURE = 1

# load ascii text and covert to lowercase
encoded_midi = processor.encode_midi(PROMPTS_PATH)

prompt_size = 100
gen_size = 2000
rand_start = np.random.randint(0, len(encoded_midi)-prompt_size)
prompt = encoded_midi[rand_start:rand_start+prompt_size]
pattern = prompt.copy()

NUM_EPOCHS = 64
TRAIN_SPLIT = 0.8
SEQ_LENGTH = 100
LAYERS = 3
HIDDEN_SIZE = 256
DROPOUT_CHANCE = 0.2
NUM_HEADS = 1
EMBED_SIZE = 1
composer = model.Composer(
    num_notes=num_vocab,
    emb_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    hidden_size=HIDDEN_SIZE,
    num_layers=LAYERS,
    dropout_chance=DROPOUT_CHANCE

)
#composer = model.Composer(num_vocab, layers, hidden_size, dropout)
composer.load_state_dict(best_model)
composer.eval()

print("\n-- PROMPT --")
print(f"{prompt}")
print("\n-- OUTPUT --")

output = []

with torch.no_grad():
    for i in range(gen_size):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(num_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        prediction = composer(x)

        # Model prediction to probability distribution using softmax
        prediction_probs = torch.softmax(prediction/TEMPERATURE, dim=1)
        prediction_probs = prediction_probs.squeeze().numpy()
        # Sample character index from distribution
        predicted_note = np.random.choice(len(prediction_probs), p=prediction_probs)
        # Get character by index

        print(list(prediction_probs))
        output.append(predicted_note)
        # Push generated character to memory
        pattern.append(int(prediction.argmax()))
        pattern.pop(0)
processor.decode_midi(output, SAVE_PATH)