import torch
import importlib
import numpy as np

import model
from midi_processor import processor

import utils

importlib.reload(utils)
importlib.reload(model)

utils.create_directory("results")

PATH = "models/maestro-8.pth"
SAVE_PATH = "results/maestro-8-1.mid"
PROMPTS_PATH = "data/maestro/midi_train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--3.midi"
# PROMPTS_PATH = "data/weimar/ArtPepper_Anthropology_FINAL.mid"
best_model, config = torch.load(PATH)
print(config)


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

composer = model.Composer(config["model"])
composer.load_state_dict(best_model)
composer.eval()

print("\n-- PROMPT --")
print(f"{prompt}")
print("\n-- OUTPUT --")

output = []

with torch.no_grad():
    for i in range(gen_size):
        x = np.reshape(pattern, (1, len(pattern), 1)) # / float(num_vocab)
        x = torch.tensor(x, dtype=torch.float32).int().squeeze(-1)

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
        # pattern.append(int(prediction.argmax()))
        pattern.append(predicted_note)
        pattern.pop(0)
processor.decode_midi(prompt+output, SAVE_PATH)