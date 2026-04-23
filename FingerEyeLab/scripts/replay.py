import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm


# --- 1. Path Setup & Argument Parsing ---
FILE_PATH = Path(__file__).resolve()
DIR_PATH = FILE_PATH.parent.parent  # Adjust based on your folder structure

# Add custom source path for local task modules.
source_path = DIR_PATH / "source/fingereye/fingereye/tasks/direct/fingereye"
sys.path.append(str(source_path))

from isaaclab.app import AppLauncher

# Initialize Parser
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")

# Existing args
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")

# New arg for Zarr path
parser.add_argument(
    "--zarr_path", 
    type=str, 
    default=None, 
    help="Path to the Zarr dataset (optional). Defaults to DIR_PATH/data/30_coin.zarr"
)

# Add AppLauncher args and parse
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# --- 2. Launch App (Must be before torch/gym imports) ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------- RTX Render Fix ----------------
import carb.settings

settings = carb.settings.get_settings()

settings.set("/rtx/renderer", "rtx-interactive")
settings.set("/rtx/pathtracing/enabled", False)
settings.set("/rtx/raytracing/fractionalCutoutOpacity", False)

settings.set("/rtx/raytracing/translucency/enabled", True)
settings.set("/rtx/raytracing/reflections/enable", True)

print("[RTX] fractionalCutoutOpacity disabled")
# ------------------------------------------------


# --- 3. Deferred Imports ---
import gymnasium as gym
import hydra
import torch
import zarr
from termcolor import cprint
from isaaclab_tasks.utils import parse_env_cfg

# Register custom tasks
import fingereye.tasks  # noqa: F401

# Pass hydra args to sys.argv for @hydra.main
sys.argv = [sys.argv[0]] + hydra_args


class SimulatorReplayer:
    def __init__(self, config, env, zarr_path):
        self.config = config
        self.env = env
        
        # Determine final Zarr path
        if zarr_path:
            self.zarr_file = Path(zarr_path)
        else:
            cprint("[WARN] No zarr_path provided, using default data/30_coin.zarr", "yellow")
            self.zarr_file = DIR_PATH / "data" / "0125_coin_test.zarr"

        cprint(f"📂 Loading Zarr from: {self.zarr_file}", "blue")
        
        if not self.zarr_file.exists():
            raise FileNotFoundError(f"Zarr file not found at {self.zarr_file}")
            
        self.zarr_group = zarr.open(str(self.zarr_file), mode='ra')

    def get_state_from_zarr(self):
        """
        Rename from get_action_from_zarr.
        Returns:
            all_states_np: (Total_T, state_dim)
            episode_ends: (N_demos,) - indices where each original demo ends
        """
        # Read states
        all_qpos_np = np.asarray(self.zarr_group["data/obs/current_all_joint_values"])
        coin_pos_np = np.asarray(self.zarr_group["data/obs/coin_pose"]) 
        all_states_np = np.concatenate([all_qpos_np, coin_pos_np], axis=1) # (Total_T, 47 + 3 + 4)
        
        # Read episode ends to know how to split the demos
        if "meta/episode_ends" in self.zarr_group:
            episode_ends = np.asarray(self.zarr_group["meta/episode_ends"])
        else:
            # Fallback if only 1 demo exists and no episode_ends recorded
            episode_ends = np.array([all_states_np.shape[0]])
            
        return all_states_np, episode_ends

    def run_aug_demo_level_replay(self):
        """
        Replay multiple demos in parallel across environments.
        
        Logic:
        1. Read all states and original episode ends.
        2. Calculate total output size = Total_Original_Steps * Num_Envs.
        3. Loop through each original demo:
           - Reset Env.
           - Loop through steps of this demo.
           - Apply same state to all Envs.
           - Write Obs to Zarr at calculated offsets.
        4. Write new expanded episode_ends.
        5. Expand and write actions to match observations.
        """
        save_name_map = {
            "current_rgb_images": "rgb_images",
            "current_rs_images": "rs_rgb_images",
            "current_joint_values": "all_qpos",
        }
        
        # -------- 1. Read Data --------
        all_states, original_episode_ends = self.get_state_from_zarr()
        
        n_demos = len(original_episode_ends)
        n_env = self.env.num_envs
        device = self.env.device
        
        # Calculate Total Output Size
        total_original_steps = all_states.shape[0]
        total_new_steps = total_original_steps * n_env
        
        cprint(f"[INFO] Replay: {n_demos} Demos -> {n_demos * n_env} Episodes (x{n_env} Envs)", "cyan")
        cprint(f"[INFO] Total Steps: {total_original_steps} -> {total_new_steps}", "cyan")

        # -------- 2. Prepare Zarr Groups --------
        root = self.zarr_group
        obs_group = root.require_group("data").require_group("obs")
        meta_group = root.require_group("meta")

        # Clear old obs data
        for k in list(obs_group.array_keys()):
            del obs_group[k]
        
        obs_datasets = None 
        
        # -------- 3. Main Replay Loop --------
        # Track where we are reading from (original data)
        read_start_idx = 0
        # Track where we are writing to (new expanded data) - global offset
        # The logic is: We append N_Env copies of Demo_i to the file sequentially.
        # So we just need a strictly increasing write counter.
        global_write_cursor = 0 
        
        new_episode_ends = []

        for demo_idx in range(n_demos):
            read_end_idx = original_episode_ends[demo_idx]
            
            # Slice the current demo states: (T_demo, state_dim)
            demo_states = all_states[read_start_idx : read_end_idx]
            T_demo = demo_states.shape[0]
            
            # Reset Environment for the new demo
            self.env.reset()
            
            # --- Stream steps for this demo ---
            # We want to write data such that:
            # [Env0_Demo0], [Env1_Demo0], ... [EnvN_Demo0]
            # BUT, we are generating them in parallel (Step 0 for all Envs, Step 1 for all Envs...)
            # So we calculate the write index:
            # global_write_cursor points to the start of this batch of episodes in Zarr
            # For a specific env_id at time t, index = global_write_cursor + (env_id * T_demo) + t
            
            pbar = tqdm(range(T_demo), desc=f"Replaying Demo {demo_idx+1}/{n_demos}", leave=False)
            for t in pbar:
                # Prepare state: (1, dim) -> (n_env, dim)
                state_t = demo_states[t : t + 1].astype(np.float32)
                states_np = np.repeat(state_t, n_env, axis=0)
                states_torch = torch.from_numpy(states_np).to(device=device, dtype=torch.float32)

                # Step Env
                obs, _, _, _, _ = self.env.replay_step(states_torch)
                
                # Move to CPU
                keys = list(obs.keys())
                keys_to_convert = ["third_view", "current_rs_images", "current_rgb_images"]
                for k in keys_to_convert:
                    if k not in obs:
                        continue
                        
                    x = obs[k]  # torch.Tensor
                    # Case 1: (N, H, W, 3) -> (N, 3, H, W)
                    if x.ndim == 4 and x.shape[-1] == 3:
                        obs[k] = x.permute(0, 3, 1, 2).contiguous()

                    # Case 2: (N, n_cams, H, W, 3) -> (N, n_cams, 3, H, W)
                    elif x.ndim == 5 and x.shape[-1] == 3:
                        obs[k] = x.permute(0, 1, 4, 2, 3).contiguous()
                # Initialize Datasets on first step of first demo
                obs_cpu = {k: v.detach().cpu().numpy() for k, v in obs.items()}
                
                if obs_datasets is None:
                    obs_datasets = {}
                    for k in keys:
                        data_shape = obs_cpu[k].shape[1:] # remove env dim
                        save_name = save_name_map.get(k, k)
                        
                        # Chunking: (1, ...) is usually good for random access training, 
                        # or (chunk_size, ...) for sequential reading. 
                        # Using 1 for simplicity based on your code.
                        obs_datasets[k] = obs_group.create_dataset(
                            name=save_name,
                            shape=(total_new_steps,) + data_shape,
                            chunks=(1,) + data_shape,
                            dtype=obs_cpu[k].dtype,
                            overwrite=True,
                        )

                # Write to Zarr
                # Input obs_cpu[k] is (n_env, H, W, C...)
                for env_id in range(n_env):
                    # Calculate position in the flat Zarr array
                    # Current batch starts at global_write_cursor
                    # Inside this batch, we organize by Env, then by Time
                    write_idx = global_write_cursor + (env_id * T_demo) + t
                    
                    for k in keys:
                        obs_datasets[k][write_idx] = obs_cpu[k][env_id]

            pbar.close()

            # --- Update Indices for next demo ---
            read_start_idx = read_end_idx
            
            # Update Episode Ends
            # We just wrote n_env new episodes, each of length T_demo
            current_total_len = global_write_cursor # Starting point
            for _ in range(n_env):
                current_total_len += T_demo
                new_episode_ends.append(current_total_len)
            
            # Advance the global cursor by total frames written in this batch
            global_write_cursor += (T_demo * n_env)

        # -------- 4. Save New Episode Ends --------
        new_episode_ends = np.array(new_episode_ends, dtype=np.int64)
        if "episode_ends" in meta_group:
            del meta_group["episode_ends"]
        meta_group.create_dataset("episode_ends", data=new_episode_ends, overwrite=True)
        
        cprint(f"[SUCCESS] Expanded episode_ends saved. Total episodes: {len(new_episode_ends)}", "green")

        # -------- 5. Expand Actions to Match Observations --------
        self.expand_actions_for_demo_level(original_episode_ends, n_env)
        
        cprint("✅ Finish Replay", "green")
        print(root.tree())

    def expand_actions_for_demo_level(self, original_episode_ends, n_env):
        """
        Reads original actions, splits them by demo, duplicates them for each env, 
        and writes back to 'original_actions'.
        """
        root = self.zarr_group
        actions_group = root.require_group("data").require_group("actions")
        
        if "original_actions" not in actions_group:
            cprint("[WARN] No original_actions found to duplicate.", "yellow")
            return

        raw_actions = actions_group["original_actions"][:] # (Total_Original_T, Dim)
        action_dim = raw_actions.shape[1]
        
        # Prepare new container
        total_new_len = raw_actions.shape[0] * n_env
        
        # We need to recreate the dataset
        if "original_actions" in actions_group:
            del actions_group["original_actions"]
            
        new_action_ds = actions_group.create_dataset(
            "original_actions",
            shape=(total_new_len, action_dim), # Assuming you want flattened (T*N, Dim)
            chunks=(1, action_dim),
            dtype=np.float32
        )
        
        read_start = 0
        write_cursor = 0
        
        cprint("Expanding actions...", "cyan")
        
        for end_idx in original_episode_ends:
            # 1. Get action for this demo
            demo_act = raw_actions[read_start : end_idx] # (T_demo, Dim)
            T_demo = demo_act.shape[0]
            
            # 2. Repeat for all envs
            # We need the output to be: Env0_Act, Env1_Act... 
            # Structure: [Env0_T0..Tn, Env1_T0..Tn]
            
            # (T_demo, Dim) -> (1, T_demo, Dim) -> (n_env, T_demo, Dim)
            expanded = np.repeat(demo_act[np.newaxis, :, :], n_env, axis=0)
            
            # Flatten to (n_env * T_demo, Dim)
            expanded_flat = expanded.reshape(-1, action_dim)
            
            # 3. Write
            new_action_ds[write_cursor : write_cursor + len(expanded_flat)] = expanded_flat
            
            read_start = end_idx
            write_cursor += len(expanded_flat)
            
        cprint(f"[SUCCESS] Expanded actions shape: {new_action_ds.shape}", "green")

    def run_frame_level_replay(self, reset_every_step=False):
        """
        Splits the total dataset across num_envs using a simple linear stride.
        Ensures exact index matching with NO shifting or padding artifacts.
        """
        save_name_map = {
            "current_rgb_images": "rgb_images",
            "current_rs_images": "rs_rgb_images",
            "current_joint_values": "all_qpos",
        }

        # 1. Read all states
        all_states, _ = self.get_state_from_zarr()
        total_frames = all_states.shape[0]
        n_env = self.env.num_envs
        device = self.env.device
        
        cprint(f"[INFO] Frame-level Replay: {total_frames} frames. Batch size: {n_env}", "cyan")

        # 2. Prepare Zarr (Delete old, get ready to create NEW)
        root = self.zarr_group
        obs_group = root.require_group("data").require_group("obs")
        
        # Cleanup
        existing_keys = list(obs_group.array_keys())
        for k in existing_keys:
             if k in save_name_map.values(): 
                 del obs_group[k]
        
        obs_datasets = {} 
        
        # 3. Main Loop: Linear Iteration
        # We process frames in chunks of 'n_env'.
        # Batch 0: Frames [0, 1, 2 ... n_env-1]
        # Batch 1: Frames [n_env, n_env+1 ... 2*n_env-1]
        
        # Calculate how many batches we need
        # If total=100, env=10, we need 10 batches.
        # If total=105, env=10, we need 11 batches (last one partial).
        num_batches = int(np.ceil(total_frames / n_env))

        for batch_idx in tqdm(range(num_batches), desc="Frame-level Replay"):
            # Calculate range for this batch
            start_idx = batch_idx * n_env
            end_idx = min(start_idx + n_env, total_frames)
            current_batch_size = end_idx - start_idx
            
            # Extract states for this batch
            # Shape: (current_batch_size, state_dim)
            batch_states_np = all_states[start_idx : end_idx]
            
            # --- HANDLE LAST PARTIAL BATCH ---
            # If we have fewer frames than n_env (e.g., last batch has 5 frames but we have 10 envs),
            # we must pad the input to the simulation to avoid shape mismatch errors in IsaacLab.
            # We will ignore the output of these padded envs later.
            if current_batch_size < n_env:
                padding_needed = n_env - current_batch_size
                # Pad with the last valid frame (just to keep physics happy)
                last_frame = batch_states_np[-1:]
                padding = np.repeat(last_frame, padding_needed, axis=0)
                batch_states_padded = np.concatenate([batch_states_np, padding], axis=0)
                batch_states_torch = torch.from_numpy(batch_states_padded).to(device, dtype=torch.float32)
            else:
                batch_states_torch = torch.from_numpy(batch_states_np).to(device, dtype=torch.float32)

            # Step Env
            if reset_every_step:
                self.env.reset() 
                self.env.replay_step(batch_states_torch) # Settle
                obs, _, _, _, _ = self.env.replay_step(batch_states_torch) # Capture
            else:
                obs, _, _, _, _ = self.env.replay_step(batch_states_torch)
            keys_to_convert = ["third_view", "current_rs_images", "current_rgb_images"]
            for k in keys_to_convert:
                if k not in obs:
                    continue

                x = obs[k]  # torch.Tensor

                # Case 1: (N, H, W, 3) -> (N, 3, H, W)
                if x.ndim == 4 and x.shape[-1] == 3:
                    obs[k] = x.permute(0, 3, 1, 2).contiguous()

                # Case 2: (N, n_cams, H, W, 3) -> (N, 3 * n_cams, H, W)
                elif x.ndim == 5 and x.shape[-1] == 3:
                    obs[k] = x.permute(0, 1, 4, 2, 3).contiguous()
            # Move to CPU
            obs_cpu = {key: val.detach().cpu().numpy() for key, val in obs.items()}
            
            # --- Initialize Zarr Datasets (Once) ---
            if len(obs_datasets) == 0:
                for key, val_batch in obs_cpu.items():
                    data_shape = val_batch.shape[1:] 
                    save_name = save_name_map.get(key, key)
                    
                    obs_datasets[key] = obs_group.create_dataset(
                        save_name, 
                        shape=(total_frames,) + data_shape, 
                        chunks=(1,) + data_shape, 
                        dtype=val_batch.dtype,
                        overwrite=True
                    )
            
            # --- Write to Disk ---
            # We write exactly 'current_batch_size' items.
            # If we padded the input, we simply slice obs_cpu[:current_batch_size] to discard garbage.
            
            write_range = slice(start_idx, end_idx)
            
            for key, dataset in obs_datasets.items():
                data_batch = obs_cpu[key] # (n_env, ...)
                
                # Take only the valid frames (ignoring any padding we added for the sim)
                valid_data = data_batch[:current_batch_size]
                
                # Bulk write (Zarr supports slice assignment)
                dataset[write_range] = valid_data

        cprint("✅ Frame Level Replay Done (Linear Strategy)", "green")


@hydra.main(config_path=str(DIR_PATH / "configs"), config_name="coin_flipping", version_base="1.1")
def main(config):
    # 1. Parse Config
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )

    # 2. Create Environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # 3. Initialize Replayer (pass the Zarr path from CLI args)
    replayer = SimulatorReplayer(
        env=env.unwrapped, 
        config=config, 
        zarr_path=args_cli.zarr_path
    )

    # 4. Select Mode based on Task Name
    if "Random" in args_cli.task:
        cprint(f"Detected 'Random' in task name: {args_cli.task}", "green")
        replayer.run_aug_demo_level_replay()
    else:
        cprint(f"Standard task name detected: {args_cli.task}", "cyan")
        replayer.run_frame_level_replay(reset_every_step=False)

    # 5. Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()