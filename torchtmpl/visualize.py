import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def show_plankton_image(img, mask):
    """
    Display an image and its mask side by side

    img is (H, W) or (H, W, C)
    mask is (H, W) or (H, W, C)
    """
    # Ensure grayscale or first channel for display
    if img.ndim == 3 and img.shape[2] == 3:
        img_display = img
        cmap = None
    else:
        img_display = img.squeeze()
        cmap = "gray"

    plt.subplot(1, 2, 1)
    plt.imshow(img_display, cmap=cmap)
    plt.title("Image")
    plt.axis("off")

    mask_display = mask.squeeze()
    plt.subplot(1, 2, 2)
    plt.imshow(mask_display, interpolation="none", cmap="tab20c")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("plankton_sample.png", bbox_inches="tight", dpi=300)
    plt.show()


def extract_patch_from_ppm(ppm_path, row_idx, col_idx, patch_size):
    """
    Extract a patch from a P5 (grayscale) or P6 (RGB) PPM image

    We determine channels based on the PPM magic number:
      P5 -> grayscale (1 channel)
      P6 -> RGB (3 channels)
    """

    with open(ppm_path, "rb") as f:
        # Read the magic number (e.g., P5 or P6)
        magic = f.readline().strip()
        # Skip comments
        line = f.readline().decode("utf-8")
        while line.startswith("#"):
            line = f.readline().decode("utf-8")

        ncols, nrows = map(int, line.split())
        maxval = int(f.readline().decode("utf-8"))

        if magic == b'P5':
            channels = 1
        elif magic == b'P6':
            channels = 3
        else:
            raise ValueError(f"Unsupported PPM format: {magic}")

        if maxval == 255:
            nbytes_per_pixel = 1
            dtype = np.uint8
        elif maxval == 65535:
            nbytes_per_pixel = 2
            dtype = np.dtype("uint16").newbyteorder(">")
        else:
            raise ValueError(f"Unsupported maxval {maxval}")

        first_pixel_offset = f.tell()
        f.seek(0, 2)  # Seek to end of file
        file_size = f.tell()
        data_size = file_size - first_pixel_offset

        # Expected size
        expected_data_size = ncols * nrows * channels * nbytes_per_pixel

        print(f"DEBUG: {ppm_path}")
        print(f"DEBUG: magic={magic}, channels={channels}, nrows={nrows}, ncols={ncols}, maxval={maxval}")
        print(f"DEBUG: expected_data_size={expected_data_size}, actual_data_size={data_size}")

        if data_size != expected_data_size:
            raise ValueError(
                f"File size mismatch for {ppm_path}. "
                f"Expected {expected_data_size}, got {data_size}"
            )

        f.seek(first_pixel_offset)  # back to pixel start
        patch_h, patch_w = patch_size

        # Check if patch fits
        if (row_idx + patch_h) > nrows or (col_idx + patch_w) > ncols:
            raise ValueError(
                f"Requested patch {patch_h}x{patch_w} at ({row_idx},{col_idx}) does not fit in image {nrows}x{ncols}"
            )

        # Allocate patch
        if channels == 1:
            patch = np.zeros((patch_h, patch_w), dtype=dtype)
        else:
            patch = np.zeros((patch_h, patch_w, channels), dtype=dtype)

        for i in range(patch_h):
            row_offset = ((row_idx + i) * ncols * channels + (col_idx * channels)) * nbytes_per_pixel
            f.seek(first_pixel_offset + row_offset, 0)
            row_data = f.read(patch_w * channels * nbytes_per_pixel)
            row_array = np.frombuffer(row_data, dtype=dtype)
            if channels == 1:
                patch[i, :] = row_array
            else:
                patch[i, :, :] = row_array.reshape(patch_w, channels)

    print(f"DEBUG: Extracted patch shape: {patch.shape}")
    return patch


if __name__ == "__main__":
    # Example paths
    scan_ppm_path = "/mounts/Datasets3/2024-2025-ChallengePlankton/train/rg20090218_scan.png.ppm"
    mask_ppm_path = "/mounts/Datasets3/2024-2025-ChallengePlankton/train/rg20090218_mask.png.ppm"

    row_idx = 1500
    col_idx = 3200
    patch_size = (4000, 8000)

    # Show full image size using PIL
    img = Image.open(scan_ppm_path)
    img_scan = np.array(img)  # Convert PIL image to a NumPy array
    img = Image.open(mask_ppm_path)
    img_mask = np.array(img)  # Convert PIL image to a NumPy array

    print(f"Full scan image shape: {img_scan.shape}")
    print(f"Full mask image shape: {img_mask.shape}")

    # Extract patches
    print("Extracting scan patch...")
    scan_patch = extract_patch_from_ppm(scan_ppm_path, row_idx, col_idx, patch_size)
    print("Extracting mask patch...")
    mask_patch = extract_patch_from_ppm(mask_ppm_path, row_idx, col_idx, patch_size)

    # Display them
    show_plankton_image(scan_patch, mask_patch)
