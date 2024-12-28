# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
import tqdm


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False



def train(model, loader, f_loss, optimizer, device, dynamic_display=True):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    from tqdm import tqdm  # Ensure tqdm is imported

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    for i, (inputs, targets) in (pbar := tqdm(enumerate(loader), total=len(loader))):
        inputs, targets = inputs.to(device), targets.to(device)

        # Ensure targets have shape (B, H, W) and type long
        targets = targets.squeeze(1).long()

        # Compute the forward propagation
        outputs = model(inputs)  # (B, 2, H, W) should be 1 channel

        loss = f_loss(outputs, targets)  # CrossLossEntropy expects (B, 2, H, W) and (B, H, W)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        pbar.set_description(f"Train loss : {total_loss/num_samples:.2f}")

    return total_loss / num_samples



def test(model, loader, f_loss, device):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    num_samples = 0
    with torch.no_grad():
        for (inputs, targets) in loader:

            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze(1).long()

            # Compute the forward propagation
            outputs = model(inputs)

            loss = f_loss(outputs, targets)

            # Update the metrics
            # We here consider the loss is batch normalized
            total_loss += inputs.shape[0] * loss.item()
            num_samples += inputs.shape[0]

    return total_loss / num_samples






def binary_list_to_string(binary_list, num_bits=6, offset=48):
    """
    Convert a list of binary digits (0s and 1s) into a string, where every `num_bits` bits represent a character.

    Arguments:
        binary_list: List of integers (0 or 1) representing binary digits.
        num_bits: Number of bits used for encoding a single character.
        offset: Offset to add to the integer representation of the binary list.

    Returns:
        Encoded string representation of the binary input.
    """
    # Ensure the binary list length is a multiple of num_bits
    padding_length = (num_bits - (len(binary_list) % num_bits)) % num_bits
    binary_list.extend([0] * padding_length)  # Add padding to make it divisible by num_bits

    # Convert chunks of `num_bits` into characters
    chars = [
        chr(offset + int("".join(map(str, binary_list[i:i + num_bits])), 2))
        for i in range(0, len(binary_list), num_bits)
    ]

    return "".join(chars)


def array_to_string(arr, num_bits=6, offset=48):
    """
    Convert a 1D NumPy array of binary predictions into an encoded string.

    Arguments:
        arr: 1D NumPy array of 0s and 1s (binary predictions).
        num_bits: Number of bits used for encoding a single character.
        offset: Offset to add to the integer representation during encoding.

    Returns:
        Encoded string representation of the array.
    """
    binary_list = arr.tolist()  # Flatten the array into a list
    return binary_list_to_string(binary_list, num_bits=num_bits, offset=offset)

