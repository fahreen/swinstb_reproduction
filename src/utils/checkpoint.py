"""
Save and load training checkpoints.

A checkpoint contains:
    - model state_dict
    - optimizer state_dict
    - AMP scaler state_dict
    - epoch number
    - best validation loss
    - patience counter

Functions:
    save_checkpoint(model, optimizer, scaler, epoch, metrics, path) -> None
    load_checkpoint(path, model, optimizer, scaler) -> dict
        Returns epoch + metrics so training can resume.
"""

# TODO: implement
