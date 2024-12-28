import os
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO)

def extract_patch_from_ppm(ppm_path, row_idx, col_idx, patch_size):
    """
    Extract a grayscale patch from a P5 PPM image.

    Arguments:
    - ppm_path: path to the PPM image
    - row_idx: row index of the patch's top-left corner
    - col_idx: column index of the patch's top-left corner
    - patch_size: (height, width) of the patch

    Returns:
    - patch: the extracted patch as a 2D numpy array
    """
    with open(ppm_path, "rb") as f:
        # Read magic number (should be P5)
        magic = f.readline().strip()
        if magic != b'P5':
            raise ValueError(f"Expected P5 format, got {magic}")

        # Skip comments
        line = f.readline().decode("utf-8")
        while line.startswith("#"):
            line = f.readline().decode("utf-8")

        # Now line should have width and height
        ncols, nrows = map(int, line.split())
        maxval = int(f.readline().decode("utf-8"))

        # Determine bit depth
        if maxval == 255:
            nbytes_per_pixel = 1
            dtype = np.uint8
        elif maxval == 65535:
            nbytes_per_pixel = 2
            dtype = np.dtype("uint16").newbyteorder(">")
        else:
            raise ValueError(f"Unsupported maxval {maxval}")

        # Verify file size
        first_pixel_offset = f.tell()
        f.seek(0, 2)  # end of file
        data_size = f.tell() - first_pixel_offset
        expected_data_size = ncols * nrows * nbytes_per_pixel
        if data_size != expected_data_size:
            raise ValueError(
                f"File size mismatch for {ppm_path}. "
                f"Expected {expected_data_size}, got {data_size}"
            )

        # Check patch bounds
        patch_h, patch_w = patch_size
        if (row_idx + patch_h) > nrows or (col_idx + patch_w) > ncols:
            raise ValueError(
                f"Requested patch {patch_h}x{patch_w} at ({row_idx},{col_idx}) "
                f"does not fit in image {nrows}x{ncols}"
            )

        f.seek(first_pixel_offset)

        # Extract patch row by row
        patch = np.zeros((patch_h, patch_w), dtype=dtype)
        for i in range(patch_h):
            row_offset = ((row_idx + i) * ncols + col_idx) * nbytes_per_pixel
            f.seek(first_pixel_offset + row_offset, 0)
            row_data = f.read(patch_w * nbytes_per_pixel)
            patch[i] = np.frombuffer(row_data, dtype=dtype)

    return patch


class PlanktonDataset(Dataset):
    """
    Dataset that loads scan and mask pairs from P5 (grayscale) PPM files.
    Scans: 8-bit (maxval=255)
    Masks: 16-bit (maxval=65535)
    Extracts all non-overlapping patches from the images.
    """

    def __init__(self, root_dir, patch_size, transform_scan=None, transform_mask=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform_scan = transform_scan
        self.transform_mask = transform_mask

        all_files = os.listdir(root_dir)
        # Identify scan and mask files by their suffix
        scan_files = [f for f in all_files if "scan.png.ppm" in f]
        mask_files = [f for f in all_files if "mask.png.ppm" in f]

        scan_files.sort()
        mask_files.sort()

        def base_name(fname):
            if "_scan.png.ppm" in fname:
                return fname.replace("_scan.png.ppm", "")
            elif "_mask.png.ppm" in fname:
                return fname.replace("_mask.png.ppm", "")
            return fname

        mask_dict = {base_name(m): m for m in mask_files}

        self.pairs = []
        for s in scan_files:
            b = base_name(s)
            if b in mask_dict:
                self.pairs.append((s, mask_dict[b]))

        assert len(self.pairs) > 0, "No scan/mask pairs found."

        # Precompute all patch coordinates for each image
        self.patches = []
        for idx, (scan_file, mask_file) in enumerate(self.pairs):
            scan_path = os.path.join(self.root_dir, scan_file)
            mask_path = os.path.join(self.root_dir, mask_file)
            nrows_s, ncols_s = self.get_image_size(scan_path)
            nrows_m, ncols_m = self.get_image_size(mask_path)
            patch_h, patch_w = self.patch_size

            # Ensure both images have compatible sizes
            max_row = min(nrows_s, nrows_m) - patch_h
            max_col = min(ncols_s, ncols_m) - patch_w

            if max_row < 0 or max_col < 0:
                raise ValueError(f"Patch size {patch_h}x{patch_w} is too large for image {scan_file}")

            # Generate all non-overlapping patch coordinates
            for row_idx in range(0, max_row + 1, patch_h):
                for col_idx in range(0, max_col + 1, patch_w):
                    self.patches.append((idx, row_idx, col_idx))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_idx, row_idx, col_idx = self.patches[idx]
        scan_file, mask_file = self.pairs[img_idx]
        scan_path = os.path.join(self.root_dir, scan_file)
        mask_path = os.path.join(self.root_dir, mask_file)

        scan_patch = extract_patch_from_ppm(scan_path, row_idx, col_idx, self.patch_size)
        mask_patch = extract_patch_from_ppm(mask_path, row_idx, col_idx, self.patch_size)

        # Convert to float32 and normalize the mask
        scan_patch = scan_patch.astype(np.float32, copy=False)
        mask_patch = (mask_patch > 6).astype(np.float32, copy=False)

        # Add channel dimension: (H,W) -> (1,H,W)
        scan_patch = scan_patch[None, :, :]
        mask_patch = mask_patch[None, :, :]

        # Apply transforms if any
        if self.transform_scan:
            scan_patch = self.transform_scan(scan_patch)
        if self.transform_mask:
            mask_patch = self.transform_mask(mask_patch)

        return torch.from_numpy(scan_patch), torch.from_numpy(mask_patch)

    @staticmethod
    def get_image_size(ppm_path):
        with open(ppm_path, "rb") as f:
            magic = f.readline().strip()
            if magic != b'P5':
                raise ValueError(f"Expected P5 format, got {magic}")

            line = f.readline().decode("utf-8")
            while line.startswith("#"):
                line = f.readline().decode("utf-8")

            ncols, nrows = map(int, line.split())
            f.readline()  # Skip maxval line
        return nrows, ncols


def get_dataloaders(data_config, use_cuda):
    logging.info("  - Dataset creation")

    transform_scan = transforms.Compose([])
    transform_mask = transforms.Compose([])

    base_dataset = PlanktonDataset(
        root_dir=data_config["trainpath"],
        patch_size=tuple(data_config["patch_size"]),
        transform_scan=transform_scan,
        transform_mask=transform_mask,
    )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(data_config["valid_ratio"] * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=use_cuda,
    )

    # The patch_size is (H, W), we know channels = 1 for grayscale
    input_size = (1,) + tuple(data_config["patch_size"])  # (1, 256, 256)
    num_classes = 2

    return train_loader, valid_loader, input_size, num_classes


if __name__ == "__main__":
    data_config = {
        "trainpath": "/mounts/Datasets3/2024-2025-ChallengePlankton/train",
        "valid_ratio": 0.2,
        "batch_size": 4,
        "num_workers": 2,
        "patch_size": [256, 256],
    }

    use_cuda = torch.cuda.is_available()
    train_loader, valid_loader, input_size, num_classes = get_dataloaders(data_config, use_cuda)
    print(num_classes, input_size)

    # Test a batch
    for scans, masks in train_loader:
        print("Mask unique values:", masks.unique())
        





