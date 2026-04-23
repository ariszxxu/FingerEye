import numpy as np
from copy import deepcopy
from datetime import datetime
import os
import pickle
from termcolor import cprint

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

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(log_dir, f"recording_{timestamp}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(payload, f)

        cprint(f"Recordings saved to {save_path}, Clearing buffer.", "yellow")

        self.clear_buffer()

