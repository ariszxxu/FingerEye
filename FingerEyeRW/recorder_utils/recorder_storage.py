import os
import time 
import pickle
from datetime import datetime
import numpy as np 
from copy import deepcopy
from termcolor import cprint
from omegaconf.listconfig import ListConfig

def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    else:
        cprint(f"[MISSED DEADLINE] late by {-t_wait:.4f}s", "red", attrs=["bold"])
    return

def slice_with_list(full_array, slice_list, dim_to_slice=-1):
    slices = []
    for item in slice_list:
        if isinstance(item, int):
            # Single index -> keep shape along that dim (unsqueeze)
            slc = full_array.take(indices=[item], axis=dim_to_slice)
        elif isinstance(item, (list, tuple, ListConfig)) and len(item) == 2:
            start, end = item
            slc = full_array[(slice(None),) * dim_to_slice + (slice(start, end),)]
        else:
            raise ValueError(f"Unsupported slice spec: {item}")
        slices.append(slc)

    return np.concatenate(slices, axis=dim_to_slice)



def refill_full_list_with_slice_indices(
    full_array: np.ndarray,
    slice_list,
    partial_array: np.ndarray,
    dim_to_slice: int = -1,
):
    # Normalize dim (allow negative indexing)
    dim = dim_to_slice if dim_to_slice >= 0 else full_array.ndim + dim_to_slice
    if dim < 0 or dim >= full_array.ndim:
        raise ValueError(f"dim_to_slice {dim_to_slice} is out of bounds for array with ndim={full_array.ndim}")

    out = np.array(full_array, copy=True)

    # Helper to treat OmegaConf ListConfig like a (start, end) tuple without importing it
    def _is_range(obj):
        try:
            return hasattr(obj, "__len__") and len(obj) == 2 and not isinstance(obj, (str, bytes))
        except Exception:
            return False

    # Walk through partial_array along `dim`, assigning segments back
    pos = 0
    for item in slice_list:
        if isinstance(item, int):
            seg_len = 1
            # partial segment selector
            p_idx = [slice(None)] * partial_array.ndim
            p_idx[dim] = slice(pos, pos + seg_len)
            # destination selector (use slice(i, i+1) to match shapes)
            f_idx = [slice(None)] * out.ndim
            f_idx[dim] = slice(item, item + 1)
            out[tuple(f_idx)] = partial_array[tuple(p_idx)]
            pos += seg_len

        elif _is_range(item):
            start, end = int(item[0]), int(item[1])
            if end < start:
                raise ValueError(f"Invalid range {item}: end < start")
            seg_len = end - start
            if seg_len < 0:
                raise ValueError(f"Invalid range {item}: negative length")

            p_idx = [slice(None)] * partial_array.ndim
            p_idx[dim] = slice(pos, pos + seg_len)

            f_idx = [slice(None)] * out.ndim
            f_idx[dim] = slice(start, end)

            out[tuple(f_idx)] = partial_array[tuple(p_idx)]
            pos += seg_len

        else:
            raise ValueError(f"Unsupported slice spec: {item}")

    # Sanity check: we should have consumed exactly the partial extent
    if pos != partial_array.shape[dim]:
        raise ValueError(
            f"Partial length mismatch along dim {dim}: consumed {pos}, "
            f"but partial has {partial_array.shape[dim]}"
        )

    return out

from typing import Any

def refill_full_with_slice_indices(
    full_array,
    slice_list,
    partial_array,
    dim_to_slice: int = -1,
):
    # --- detect backend ---
    is_torch = hasattr(full_array, "dim") and hasattr(full_array, "clone")
    is_numpy = hasattr(full_array, "ndim") and hasattr(full_array, "shape") and not is_torch

    if not (is_torch or is_numpy):
        raise TypeError(f"full_array must be numpy.ndarray or torch.Tensor, got {type(full_array)}")

    if type(full_array) is not type(partial_array):
        # allow torch tensor subclass? keep simple
        if not (is_torch and hasattr(partial_array, "dim")) and not (is_numpy and hasattr(partial_array, "ndim")):
            raise TypeError(f"partial_array type mismatch: {type(full_array)} vs {type(partial_array)}")

    # --- normalize dim ---
    ndim = full_array.dim() if is_torch else full_array.ndim
    dim = dim_to_slice if dim_to_slice >= 0 else ndim + dim_to_slice
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim_to_slice {dim_to_slice} out of bounds for ndim={ndim}")

    # --- make output copy ---
    out = full_array.clone() if is_torch else full_array.copy()

    # --- helper: range-like (ListConfig/list/tuple) ---
    def _is_range2(obj: Any) -> bool:
        if isinstance(obj, (str, bytes)):
            return False
        try:
            if len(obj) != 2:
                return False
            _ = obj[0]
            _ = obj[1]
            return True
        except Exception:
            return False

    # --- size along sliced dim ---
    size_d = out.size(dim) if is_torch else out.shape[dim]
    partial_len = partial_array.size(dim) if is_torch else partial_array.shape[dim]

    # --- fill ---
    pos = 0
    for item in slice_list:
        # -------- single index --------
        if isinstance(item, int):
            idx = item
            if idx < 0:
                idx += size_d
            if idx < 0 or idx >= size_d:
                raise IndexError(f"index {item} out of range for dim size {size_d}")

            seg_len = 1
            if pos + seg_len > partial_len:
                raise ValueError(f"partial_array too short: need pos+{seg_len} <= {partial_len}, got pos={pos}")

            # partial selector
            p_idx = [slice(None)] * ndim
            p_idx[dim] = slice(pos, pos + 1)

            # destination selector (slice(idx, idx+1) keeps dim)
            f_idx = [slice(None)] * ndim
            f_idx[dim] = slice(idx, idx + 1)

            out[tuple(f_idx)] = partial_array[tuple(p_idx)]
            pos += 1
            continue

        # -------- range [start, end) --------
        if _is_range2(item):
            start, end = int(item[0]), int(item[1])

            # python-like negative handling
            if start < 0:
                start += size_d
            if end < 0:
                end += size_d

            # Keep strict bounds; change here if clamp behavior is desired.
            if start < 0 or start > size_d:
                raise IndexError(f"start {item[0]} out of range for dim size {size_d}")
            if end < 0 or end > size_d:
                raise IndexError(f"end {item[1]} out of range for dim size {size_d}")
            if end < start:
                raise ValueError(f"Invalid range {item}: end < start")

            seg_len = end - start
            if pos + seg_len > partial_len:
                raise ValueError(f"partial_array too short: need pos+{seg_len} <= {partial_len}, got pos={pos}")

            p_idx = [slice(None)] * ndim
            p_idx[dim] = slice(pos, pos + seg_len)

            f_idx = [slice(None)] * ndim
            f_idx[dim] = slice(start, end)

            out[tuple(f_idx)] = partial_array[tuple(p_idx)]
            pos += seg_len
            continue

        raise ValueError(f"Unsupported slice spec: {item} (type={type(item)})")

    # --- sanity check ---
    if pos != partial_len:
        raise ValueError(
            f"Partial length mismatch along dim {dim}: consumed {pos}, but partial has {partial_len}"
        )

    return out


def disable_and_refill(
    full_array,
    disabled_list,
    init_values = None,
    dim_to_slice = -1,
    refill_values = np.pi
):
    is_numpy = isinstance(full_array, np.ndarray)
    arr = full_array.copy() if is_numpy else full_array.clone()

    init_array = None
    if init_values is not None:
        if is_numpy:
            init_array = np.array(init_values)
        else:
            init_array = np.array(init_values)
   
    for item in disabled_list:
        indexer = [slice(None)] * arr.ndim
        if isinstance(item, int):
            indexer[dim_to_slice] = item
            if init_array is None:
                fill = refill_values
            else:
                fill = init_array[item]  
            arr[tuple(indexer)] = fill

        elif isinstance(item, (list, tuple, ListConfig)) and len(item) == 2:
            start, end = item
            indexer[dim_to_slice] = slice(start, end)
            if init_array is None:
                fill = refill_values
            else:
                fill = init_array[start:end]
            arr[tuple(indexer)] = fill

        else:
            raise ValueError(f"Unsupported slice spec: {item}")
    return arr

class RecorderStorage:
    def __init__(self):
        self.buffer_dict = {}

    def clear_buffer(self):
        self.buffer_dict = {}

    def append_buffer(
        self, data_dict,
    ):
        for k, v in data_dict.items():
            if k not in self.buffer_dict:
                self.buffer_dict[k] = []
            self.buffer_dict[k].append(deepcopy(v))

    def save_recordings(self, other_payload=None):
        if len(self.buffer_dict) == 0:
            cprint("No recordings to save.", "yellow")
            return
        
        payload = self.buffer_dict

        for k, v in payload.items():
            if k =='aux_rs':
                payload[k] = np.concatenate(v, axis=0)
            else:
                payload[k] = np.stack(v, axis=0)

        if other_payload:
            payload.update(other_payload)

        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("logs", f"recording_{timestamp}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(payload, f)

        cprint(f"Recordings saved to {save_path}, Clearing buffer.", "yellow")

        self.clear_buffer()


if __name__ == "__main__":

    # Simple test
    arr = np.arange(10)

    slice_list = [[7, 11], [19, 23]]
    sliced = slice_with_list(arr, slice_list, dim_to_slice=-1)
    print("Sliced array:\n", sliced)

    refilled = refill_full_list_with_slice_indices(arr, slice_list, sliced, dim_to_slice=-1)
    print("Refilled array:\n", refilled)

    assert np.array_equal(arr, refilled), "Refilled array does not match original!"