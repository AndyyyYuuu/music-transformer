config = {
    "architecture": "transformer",
    "num_epochs": 64,
    "train_split": 0.8,
    "batch_size": 256,
    "learning_rate": 0.005,
    "data": {
        "name": "GiantMIDI",
        "train_split": 0.8,
        "batch_size": 128,
        "vocab": None,
        "subset_prop": 0.05,
        "train_set_size": 800,
        "valid_set_size": 200
    },
    "model": {
        "architecture": "transformer",
        "sequence_length": 1024,
        "layers": 3,
        "hidden_size": 512,
        "dropout_chance": 0.2,
        "attention_heads": 6,
        "embed_size": 80
    },
    "training_info": {
        "min_loss": None,
        "epoch_at": None
    },
    "model_id": 10

}