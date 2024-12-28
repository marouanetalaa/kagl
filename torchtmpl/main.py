# coding: utf-8

# Standard imports
import csv
import logging
import sys
import os
import pathlib

# External imports
import numpy as np
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo

# Local imports
from . import data_test
from . import data
from . import models
from . import optim
from . import utils

def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders for segmentation
    # get_dataloaders should return train_loader, valid_loader, input_size, num_classes
    # input_size should be (C, H, W)
    # num_classes should be 2 for binary segmentation
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    assert num_classes == 2, "num_classes should be 2 for binary segmentation."

    # Build the model (expecting input_size = (C, H, W))
    logging.info("= Model")
    model_config = config["model"]
    # Do not slice input_size again, just pass it as is.
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss (CrossEntropyLoss for segmentation)
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"])  # Should return an nn.CrossEntropyLoss or similar

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Logging and checkpoint setup
    logging_config = config["logging"]
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Save the config
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Prepare a summary
    # Obtain one batch from train_loader to know the exact input shape (B, C, H, W)
    batch = next(iter(train_loader))
    batch_scans, batch_masks = batch
    # For model summary, we need (B, C, H, W), so input_shape = batch_scans.shape
    # torchinfo.summary requires the full batch size as well, so pass batch_scans.shape directly.
    input_shape = batch_scans.shape

    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_shape)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Model checkpointing
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    # Training loop
    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train(model, train_loader, loss, optimizer, device)
        # Validate
        test_loss = utils.test(model, valid_loader, loss, device)

        updated = model_checkpoint.update(test_loss)
        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Log metrics
        metrics = {"train_CE": train_loss, "test_CE": test_loss}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)



######################################################
#  Helper functions for 6-bit offset(48) encoding
######################################################
def binary_list_to_string(binary_list, num_bits=6, offset=48):
    """
    Convert a list of binary digits (0s and 1s) into a string, 
    where every `num_bits` bits represent one character 
    with a given ASCII offset.
    """
    if len(binary_list) % num_bits != 0:
        raise ValueError(f"binary_list length must be multiple of {num_bits}.")

    chars = []
    for i in range(0, len(binary_list), num_bits):
        chunk = binary_list[i : i + num_bits]
        val = offset + int("".join(map(str, chunk)), 2)
        chars.append(chr(val))

    return "".join(chars)


def array_to_string(arr: np.ndarray, num_bits=6, offset=48):
    """
    Flattens a 1D or 2D array of 0/1 values into a single string
    using 6-bit encoding and an ASCII offset of 48.
    """
    # If 2D row, flatten it, else assume 1D row
    flattened = arr.ravel().tolist()

    # Pad to make length multiple of num_bits
    pad_len = (num_bits - len(flattened) % num_bits) % num_bits
    flattened += [0] * pad_len

    return binary_list_to_string(flattened, num_bits=num_bits, offset=offset)


######################################################
#  The main test function
######################################################
def test(config):
    """
    Perform testing and create a submission file.
    For test, we have a test_loader that provides only scans (B,1,H,W).
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Load the test data
    logging.info("= Loading the test data")
    test_loader = data_test.get_test_loader(config["data"], use_cuda)

    logging.info("= Loading the trained model")
    model_config = config["model"]

    # Determine input size from the first batch of test loader
    first_batch = next(iter(test_loader))
    scans, _ = first_batch
    input_channels = 1  # Grayscale
    height, width = scans.shape[2], scans.shape[3]
    num_classes = 2  # Binary segmentation

    # Build and load the model
    model = models.build_model(model_config, (input_channels, height, width), num_classes)
    model.to(device)

    checkpoint_path = config["test"]["checkpoint"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logging.info("= Generating predictions")
    # Dictionary {scan_filename: [pred_mask_row0, pred_mask_row1, ...]}
    predictions = {}

    with torch.no_grad():
        for scans, metadata in test_loader:
            # scans: shape (B,1,H,W)
            scans = scans.to(device)
            outputs = model(scans)  # shape (B, 1, H, W)
            preds = (outputs.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)  # binary 0/1

            # metadata is typically (filenames, row_idxs, col_idxs, ...)
            # for a typical line: (scan_file, row_idx, col_idx)
            for i, (scan_file, row_idx, col_idx) in enumerate(zip(*metadata)):
                if scan_file not in predictions:
                    predictions[scan_file] = []
                # preds[i] is a full row or slice of the final mask
                predictions[scan_file].append(preds[i])

    # Write submission
    logging.info("= Writing the submission file")
    submission_path = config["test"]["submission_file"]
    write_submission(predictions, submission_path)
    logging.info(f"Submission file written to {submission_path}")


def write_submission(predictions, submission_file, num_bits=6, offset=48):
    """
    Write predictions to a CSV file in the same format as the sample:
    Header: "Id,Target"
    Each line:  filename_rowIdx,"ASCII-encoded mask row"

    predictions: dict
       Keys = filename
       Values = list of 2D arrays (or 1D rows) that represent predicted masks
    """
    with open(submission_file, "w", newline="") as f:
        f.write("Id,Target\n")

        for filename, rows in predictions.items():
            # rows is a list of 2D (or 1D) arrays
            # For typical usage, each element might be one row of the image mask.
            for row_idx, row in enumerate(rows):
                encoded = array_to_string(row, num_bits=num_bits, offset=offset)
                # Write in the form: filename_rowIdx,"encoded"
                f.write(f"{filename}_{row_idx},\"{encoded}\"\n")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
