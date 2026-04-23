from pynput import keyboard
from threading import Lock
import time
from termcolor import cprint

class KeyboardActionManager:
    def __init__(self):
        self.key_to_action = {
            "k": "z-",
            "j": "z+",
            "a": "x+",
            "d": "x-",
            "s": "y+",
            "w": "y-",
            "enter": "save buffer",
            "\\": "clear buffer", 
            "-": "Recording Enabled",  
            "=": "Recording Disabled", 
            "[": "Robot Control Enabled",
            "]": "Robot Control Disabled",
            "backspace": "Init Robot",
            "up": "up",
            "down": "down",
            "0": "0",            
        }
        self._pressed = set()
        self._lock = Lock()
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()

    def _norm_key(self, key):
        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char.lower()
        if key == keyboard.Key.enter:
            return "enter"
        if key == keyboard.Key.backspace:
            return "backspace"
        if key == keyboard.Key.up:
            return "up"
        if key == keyboard.Key.down:
            return "down"
        return None

    def _on_press(self, key):
        k = self._norm_key(key)
        if k:
            with self._lock:
                self._pressed.add(k)

    def _on_release(self, key):
        k = self._norm_key(key)
        if k:
            with self._lock:
                self._pressed.discard(k)

    def get_current_actions(self):
        with self._lock:
            return [
                action for key, action in self.key_to_action.items()
                if key in self._pressed
            ]

    def stop(self):
        self._listener.stop()


if __name__ == "__main__":
    mgr = KeyboardActionManager()
    print("Hold keys (q,e,w,s,a,d,enter,backspace). Press Esc to quit.")

    try:
        while True:
            actions = mgr.get_current_actions()
            if actions:
                print("Current actions:", actions)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    mgr.stop()
