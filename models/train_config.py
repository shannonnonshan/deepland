MID_CONFIG = {
    "data_root": "datasets",
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "seed": 42,
    "output_root": "outputs_midterm",
}

FINAL_CONFIG = {
    "epochs": 200,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "num_workers": 2,
    "seed": 42,
    "output_root": "outputs_final",
}

VERIFY_CONFIG = {
    "batch_size": 256,
    "num_workers": 2,
}
