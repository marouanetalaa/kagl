import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def read_ppm_header(ppm_path):
    with open(ppm_path, "rb") as f:
        magic = f.readline().strip()
        if magic != b'P5':
            raise ValueError(f"Expected P5 format for test scans, got {magic}")

        # Skip comments
        line = f.readline().decode("utf-8")
        while line.startswith("#"):
            line = f.readline().decode("utf-8")

        ncols, nrows = map(int, line.split())
        maxval = int(f.readline().decode("utf-8"))

    return nrows, ncols, maxval

def extract_patch_from_ppm(ppm_path, row_idx, col_idx, patch_size):
    with open(ppm_path, "rb") as f:
        magic = f.readline().strip()
        if magic != b'P5':
            raise ValueError(f"Expected P5 format, got {magic}")

        # Skip comments
        line = f.readline().decode("utf-8")
        while line.startswith("#"):
            line = f.readline().decode("utf-8")

        ncols, nrows = map(int, line.split())
        maxval = int(f.readline().decode("utf-8"))

        if maxval == 255:
            nbytes_per_pixel = 1
            dtype = np.uint8
        elif maxval == 65535:
            nbytes_per_pixel = 2
            dtype = np.dtype("uint16").newbyteorder(">")
        else:
            raise ValueError(f"Unsupported maxval {maxval}")

        first_pixel_offset = f.tell()
        f.seek(0, 2)  # end
        data_size = f.tell() - first_pixel_offset
        expected_data_size = ncols * nrows * nbytes_per_pixel
        if data_size != expected_data_size:
            raise ValueError(f"File size mismatch for {ppm_path}")

        patch_h, patch_w = patch_size
        if (row_idx + patch_h) > nrows or (col_idx + patch_w) > ncols:
            raise ValueError("Requested patch does not fit in the image.")

        f.seek(first_pixel_offset)
        patch = np.zeros((patch_h, patch_w), dtype=dtype)
        for i in range(patch_h):
            row_offset = ((row_idx + i) * ncols + col_idx) * nbytes_per_pixel
            f.seek(first_pixel_offset + row_offset, 0)
            row_data = f.read(patch_w * nbytes_per_pixel)
            patch[i] = np.frombuffer(row_data, dtype=dtype)

    return patch

class PlanktonTestDataset(Dataset):
    """
    Test dataset:
    - Iterates over each test image.
    - Splits each into patches of size `patch_size`.
    - No randomness, fully deterministic traversal.
    - Returns scan patches (no mask) and metadata (filename, row_idx, col_idx).
    """

    def __init__(self, root_dir, patch_size, transform=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform

        # List test scan files (no masks)
        # In the test folder, we have only scans and a taxa.csv file
        self.scan_files = [f for f in os.listdir(root_dir) if f.endswith("scan.png.ppm")]
        self.scan_files.sort()

        self.patches_info = []
        # Precompute all patches for all test images
        for scan_file in self.scan_files:
            scan_path = os.path.join(root_dir, scan_file)
            nrows, ncols, maxval = read_ppm_header(scan_path)
            patch_h, patch_w = self.patch_size

            # Compute how many patches fit horizontally and vertically
            # For test, you likely want to cover the entire image. If the image size is not divisible
            # by the patch size, you could either ignore the remainder or pad the image.
            # Here we assume we just tile fully and possibly handle partial patches if needed.
            
            # Simple approach: only full patches
            n_patches_y = nrows // patch_h
            n_patches_x = ncols // patch_w

            for py in range(n_patches_y):
                for px in range(n_patches_x):
                    row_idx = py * patch_h
                    col_idx = px * patch_w
                    self.patches_info.append((scan_file, row_idx, col_idx))

    def __len__(self):
        return len(self.patches_info)

    def __getitem__(self, idx):
        scan_file, row_idx, col_idx = self.patches_info[idx]
        scan_path = os.path.join(self.root_dir, scan_file)
        
        scan_patch = extract_patch_from_ppm(scan_path, row_idx, col_idx, self.patch_size).astype(np.float32)
        # (H, W) -> (1, H, W)
        scan_patch = scan_patch[None, :, :]

        if self.transform:
            scan_patch = self.transform(scan_patch)

        # Return also some metadata so we know where this patch belongs
        return torch.from_numpy(scan_patch), (scan_file, row_idx, col_idx)

def get_test_loader(data_config, use_cuda):
    transform_scan = transforms.Compose([])
    
    test_dataset = PlanktonTestDataset(
        root_dir=data_config["testpath"],
        patch_size=tuple(data_config["patch_size"]),
        transform=transform_scan
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=use_cuda,
    )

    return test_loader

if __name__ == "__main__":
    data_config = {
        "testpath": "/mounts/Datasets3/2024-2025-ChallengePlankton/test",
        "batch_size": 4,
        "num_workers": 2,
        "patch_size": [256, 256],
    }
    use_cuda = torch.cuda.is_available()

    test_loader = get_test_loader(data_config, use_cuda)

    # Test reading some patches
    for scans, info in test_loader:
        print("Batch scans shape:", scans.shape)  # (B, 1, 256, 256)
        print("Metadata:", info)  # (scan_file, row_idx, col_idx) for each sample
        break
