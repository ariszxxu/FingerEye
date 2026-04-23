from typing import Optional
import numpy as np
import numba
from fingereye.utils.dataset.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = []
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue

        ep_start = 0 if i == 0 else episode_ends[i - 1]
        ep_end = episode_ends[i]
        episode_length = ep_end - ep_start

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + ep_start
            buffer_end_idx = min(idx + sequence_length, episode_length) + ep_start

            start_offset = buffer_start_idx - (idx + ep_start)
            end_offset = (idx + sequence_length + ep_start) - buffer_end_idx

            sample_start_idx = start_offset
            sample_end_idx = sequence_length - end_offset

            indices.append(
                [
                    buffer_start_idx,
                    buffer_end_idx,
                    sample_start_idx,
                    sample_end_idx,
                    ep_start,
                    ep_end,
                ]
            )

    return np.asarray(indices, dtype=np.int64)

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        # train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        # train_idxs = curr_train_idxs[train_idxs_idx]
        # just choose the first n_train indices for reproducibility
        train_idxs = curr_train_idxs[:n_train]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

def _safe_fill_value(dtype):
    """Return a valid sentinel for the given dtype."""
    if np.issubdtype(dtype, np.floating):
        return np.nan
    elif np.issubdtype(dtype, np.unsignedinteger):
        # use the max representable value (e.g., 255 for uint8)
        return np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.signedinteger):
        return -1
    elif np.issubdtype(dtype, np.bool_):
        return False
    else:
        # fallback: zero
        return 0
    
class SequenceSampler:
    def __init__(
        self,
        replay_buffer,
        sequence_length,
        pad_before=0,
        pad_after=0,
        keys=None,
        key_first_k=None,
        key_slice=None,
        key_sample_t=None,
        episode_mask=None,
    ):
        assert sequence_length >= 1

        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        self.keys = list(keys) if keys is not None else list(replay_buffer.keys())

        self.key_first_k = key_first_k or {}
        self.key_slice = key_slice or {}

        # preconvert offsets to numpy once
        self.key_sample_t = {
            k: np.asarray(v, dtype=np.int64)
            for k, v in (key_sample_t or {}).items()
        }

        # detect contiguous offsets once
        self.key_sample_t_contiguous = {}
        for k, offs in self.key_sample_t.items():
            self.key_sample_t_contiguous[k] = (
                offs.size > 1 and np.all(offs[1:] == offs[:-1] + 1)
            )

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones_like(episode_ends, dtype=bool)

        if np.any(episode_mask):
            self.indices = create_indices(
                episode_ends,
                sequence_length,
                episode_mask,
                pad_before,
                pad_after,
            )
        else:
            self.indices = np.zeros((0, 6), dtype=np.int64)


    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            ep_start,
            ep_end,
        ) = self.indices[idx]

        result = {}

        for key in self.keys:
            input_arr = self.replay_buffer[key]

            # -------------------------------------------------
            # Case 0: key_sample_t (optimized)
            # -------------------------------------------------
            if key in self.key_sample_t:
                offsets = self.key_sample_t[key]
                anchor_idx = buffer_start_idx - sample_start_idx

                # fresh target every sample (no in-place mutation)
                target = anchor_idx + offsets
                target = np.clip(target, ep_start, ep_end - 1)

                # Only slice if target is strictly consecutive after clamping
                # This guarantees output length == len(offsets)
                if (
                    target.size >= 2
                    and (target[-1] - target[0] + 1) == target.size
                    and np.all(target[1:] == target[:-1] + 1)
                ):
                    result[key] = input_arr[target[0] : target[-1] + 1]
                else:
                    result[key] = input_arr[target]

                continue



            n_data = buffer_end_idx - buffer_start_idx

            # -------------------------------------------------
            # Case 1: key_slice
            # -------------------------------------------------
            if key in self.key_slice:
                rel = np.asarray(self.key_slice[key], dtype=np.int64)
                rel = np.clip(rel, 0, n_data - 1)
                result[key] = input_arr[buffer_start_idx + rel]
                continue

            # -------------------------------------------------
            # Case 2: key_first_k
            # -------------------------------------------------
            if key in self.key_first_k:
                k = min(int(self.key_first_k[key]), self.sequence_length)
                shape_tail = input_arr.shape[1:]
                out = np.empty((k,) + shape_tail, dtype=input_arr.dtype)

                first = input_arr[buffer_start_idx]
                last = input_arr[buffer_end_idx - 1]
                out[:] = first

                a = max(sample_start_idx, 0)
                b = min(sample_end_idx, k)
                if a < b:
                    buf0 = buffer_start_idx + (a - sample_start_idx)
                    buf1 = buffer_start_idx + (b - sample_start_idx)
                    out[a:b] = input_arr[buf0:buf1]

                if sample_end_idx < k:
                    out[sample_end_idx:k] = last

                result[key] = out
                continue

            # -------------------------------------------------
            # Case 3: full padded window
            # -------------------------------------------------
            data = input_arr[buffer_start_idx:buffer_end_idx]

            if sample_start_idx > 0 or sample_end_idx < self.sequence_length:
                padded = np.empty(
                    (self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                padded[:] = data[0]
                padded[sample_start_idx:sample_end_idx] = data
                padded[sample_end_idx:] = data[-1]
                data = padded

            result[key] = data

        return result
