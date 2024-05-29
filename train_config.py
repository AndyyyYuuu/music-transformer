config = {
    "architecture": "transformer",
    "num_epochs": 64,
    "train_split": 0.8,
    "batch_size": 32,
    "data": {
        "name": "MAESTRO",
        "train_split": 0.8,
        "batch_size": 32,
        "vocab": None
    },
    "model": {
        "architecture": "transformer",
        "sequence_length": 1024,
        "layers": 3,
        "hidden_size": 512,
        "dropout_chance": 0.2,
        "attention_heads": 4,
        "embed_size": 64
    },
    "training_info": {
        "min_loss": None,
        "epoch_at": None
    },
    "model_id": 8

}