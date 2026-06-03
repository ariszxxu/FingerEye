# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Final FingerEye release task registrations."""

import gymnasium as gym

from . import agents


def _register_direct_task(task_id: str, env_module: str, env_class: str, cfg_module: str, cfg_class: str) -> None:
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.{env_module}:{env_class}",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{cfg_module}:{cfg_class}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
            "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        },
    )


_register_direct_task(
    "FingerEye-CS-Lab-Direct-v1",
    env_module="fingereye_cs_env",
    env_class="FingerEyeCSLabEnv",
    cfg_module="fingereye_cs_env_cfg",
    cfg_class="FingerEyeCSLabEnvCfg",
)

_register_direct_task(
    "FingerEye-CIH-Lab-Direct-v1",
    env_module="fingereye_cih_env",
    env_class="FingerEyeCIHLabEnv",
    cfg_module="fingereye_cih_env_cfg",
    cfg_class="FingerEyeCIHLabEnvCfg",
)

_register_direct_task(
    "FingerEye-PN-Lab-Direct-v1",
    env_module="fingereye_pn_env",
    env_class="FingerEyePNLabEnv",
    cfg_module="fingereye_pn_env_cfg",
    cfg_class="FingerEyePNLabEnvCfg",
)
