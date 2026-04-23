from pathlib import Path
from typing import Union
from termcolor import cprint
import os


def output_path_confirmation(save_path_str: Union[Path, str], only_warn: bool = False):
    if isinstance(save_path_str, Path):
        save_path_str = str(save_path_str.resolve())
    if os.path.exists(save_path_str):
        cprint("Data already exists at {}".format(save_path_str), "red")
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        if not only_warn:
            user_input = input()
        else:
            user_input = "y"
        if user_input == "y":
            cprint("Overwriting {}".format(save_path_str), "red")
            os.system("rm -rf {}".format(save_path_str))
        else:
            cprint("Exiting", "red")
            return
