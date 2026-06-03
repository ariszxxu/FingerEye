# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FingerEye Lab namespace package."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

try:
    # Register Gym environments when Isaac Lab task discovery is available.
    from .tasks import *
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab_tasks":
        raise

try:
    # Register UI extensions when running inside Isaac Sim.
    from .ui_extension_example import *
except ModuleNotFoundError as exc:
    if not str(exc.name).startswith("omni"):
        raise
