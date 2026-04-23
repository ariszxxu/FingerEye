from datetime import datetime
from typing import List, Tuple

import numpy as np
import zarr
import torch
from fingereye.models.vision.radio import RADIO
from numcodecs import Blosc
import argparse

def get_chunk_size(array: np.ndarray):
    """
    Determine appropriate chunk size based on array dimensions.

    Parameters:
    - array: Numpy array for which to calculate chunk size.

    Returns:
    - A tuple representing the chunk size.
    """
    if array.ndim >= 2:  
        shape = array.shape
        chunk_shape = [1] + list(shape[1:])
        return tuple(chunk_shape)
    else:
        return None
        
def _to_bchw_and_dims(
    np_batch: np.ndarray,
    input_order: str,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Assumes images are channels-last:
      (B, NV, 3, H, W) or (B, 3, H, W) -> returns (B*NV, 3, H, W)
    Returns (bchw, (B, NV, H, W))
    """
    if np_batch.ndim == 5:
        B, NV, C, H, W = np_batch.shape
        arr = np_batch
    elif np_batch.ndim == 4:
        B, C, H, W = np_batch.shape
        NV = 1
        arr = np_batch.reshape(B, 1, C, H, W)
    else:
        raise ValueError(f"Expected 4D or 5D channels-last arrays, got {np_batch.shape}")

    if input_order.lower() == "bgr":
        arr = arr[:, :, [2, 1, 0]]  # BGR -> RGB

    arr = arr.astype(np.float32) / 255.0
    # (B, NV, 3, H, W) -> (B*NV, 3, H, W)
    if arr.shape[-1] == 3:
        arr = arr.transpose(0, 1, 4, 2, 3)  # (B, NV, H, W, 3) -> (B, NV, 3, H, W)
    bchw = arr.reshape(arr.shape[0] * arr.shape[1], 3, arr.shape[3], arr.shape[4])
    return bchw, (B, NV, arr.shape[3], arr.shape[4])

def add_radio_summary_to_zarr(
    zarr_path: str,
    image_keys: List[str],  # e.g. ["data/obs/rgb_images", "data/obs/rs_rgb_images"]
    input_order: str = "bgr",
    batch_size: int = 512,
    dtype: np.dtype = np.float32,
    compressor: Blosc = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
) -> None:
    """
    For each image key in `image_keys`, compute RADIO per-view summaries and
    store them at: <key>_radio with shape (T, NV, D).

    - Keeps all other arrays/groups untouched.
    - Assumes channels-last images: (T, NV, H, W, 3) or (T, H, W, 3).
    - `input_order` describes stored channel order ("bgr" or "rgb").
    """
    root = zarr.open(zarr_path, mode="a")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RADIO().to(device).eval()

    for key in image_keys:
        if key not in root:
            raise KeyError(f"Image key '{key}' not found in {zarr_path}")
        img_arr = root[key] # (1285, 1, 3, 240, 320)
        img_h, img_w = img_arr.shape[-2], img_arr.shape[-1]
        if not isinstance(img_arr, zarr.core.Array):
            raise TypeError(f"'{key}' is not a Zarr array")

        T = img_arr.shape[0]
        if T == 0:
            continue

        # Peek to get NV and infer D
        sample_np = img_arr.get_basic_selection((slice(0, min(1, T)),))
        # Force to channels-last handling
        if sample_np.ndim == 5:
            NV_s = sample_np.shape[1]
        elif sample_np.ndim == 4:
            NV_s = 1
        else:
            raise ValueError(f"Unsupported sample shape {sample_np.shape} for key '{key}'")

        D = model.summary_dim  # RADIO has fixed summary dim

        # Output path: sibling key with suffix "_radio"
        out_key = f"{key}_radio"
        out_grid_key = f"{key}_radio_grid"

        # Create or validate output array
        if out_key in root:
            out_arr = root[out_key]
            if not isinstance(out_arr, zarr.core.Array):
                raise TypeError(f"'{out_key}' exists but is not a Zarr array")
            if out_arr.shape != (T, NV_s, D):
                raise ValueError(
                    f"Existing '{out_key}' has shape {out_arr.shape}, expected {(T, NV_s, D)}."
                )
        else:
            chunk_size = get_chunk_size(np.zeros((T, NV_s, D), dtype=dtype))
            # Ensure parent groups exist
            parent = root
            parts = out_key.split("/")
            for p in parts[:-1]:
                parent = parent.require_group(p)
            out_arr = parent.create(
                name=parts[-1],
                shape=(T, NV_s, D),
                chunks=chunk_size,
                dtype=dtype,
                compressor=None,
                overwrite=True,
            )

        # Process by time batches
        nv_batch_size = batch_size // NV_s
        with torch.inference_mode():
            for start in range(0, T, nv_batch_size):
                print(f"[{datetime.now().isoformat()}] Processing '{key}': {start} / {T}", flush=True)

                end = min(T, start + nv_batch_size)
                np_batch = img_arr.get_basic_selection((slice(start, end),))
                bchw, (B, NV, _, _) = _to_bchw_and_dims(np_batch, input_order)

                t_batch = torch.from_numpy(bchw).to(device, non_blocking=True)

                feat_grid, summary = model.get_feature_grid(t_batch, return_processed_img=False)  # (B*NV, D)

                summary_np = (
                    summary.reshape(B, NV, D)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(dtype, copy=False)
                )

                out_arr[start:end, :, :] = summary_np

    print(root.tree())
    return

# ----------------- CLI Argument Parsing ----------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RADIO embeddings for Zarr images.")

    # 1. Zarr Path (Required)
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the Zarr dataset file/directory.",
    )

    # 2. Image Keys (Optional, List)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["data/obs/rs_rgb_images", "data/obs/rgb_images"],
        help="List of image keys to process inside the Zarr file. (Default: rs_rgb_images & rgb_images)",
    )

    # 3. Input Order (Optional)
    parser.add_argument(
        "--order",
        type=str,
        default="rgb",
        choices=["rgb", "bgr"],
        help="Color channel order of the input images. (Default: rgb)",
    )

    # 4. Batch Size (Optional)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for inference. (Default: 256)",
    )

    args = parser.parse_args()

    add_radio_summary_to_zarr(
        zarr_path=args.path,
        image_keys=args.keys,
        input_order=args.order,
        batch_size=args.batch_size,
    )

