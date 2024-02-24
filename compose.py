import torch
import numpy as np

import model
from midi_processor import processor

PATH = "models/jazz-2.pth"
SAVE_PATH = "results/jazz-2-1.mid"
PROMPTS_PATH = "dataset/ArtPepper_Anthropology_FINAL.mid"

best_model, num_vocab, best_loss, epoch = torch.load(PATH)
TEMPERATURE = 1

# load ascii text and covert to lowercase
encoded_midi = processor.encode_midi(PROMPTS_PATH)

prompt_size = 100
gen_size = 500
rand_start = np.random.randint(0, len(encoded_midi)-prompt_size)
prompt = encoded_midi[rand_start:rand_start+prompt_size]
pattern = prompt.copy()

composer = model.Composer(num_vocab, 2)
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

        print(predicted_note, end=' ')
        output.append(predicted_note)
        # Push generated character to memory
        pattern.append(int(prediction.argmax()))
        pattern.pop(0)
processor.decode_midi(output, SAVE_PATH)