import os
import json
from VLABench.utils.register import register
from VLABench.envs.dm_env import LM4ManipDMEnv
from VLABench.configs import name2config
from VLABench.utils.utils import find_key_by_value
import copy
# load global robot config here, corresponding to different embodiments
with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/robot_config.json"), "r") as f:
    ROBOT_CONFIG= json.load(f)

with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task_config.json"), "r") as f:
    TASK_CONFIG = json.load(f)

def load_env(task, robot="franka", config=None, time_limit=float('inf'), reset_wait_step=100, **kwargs):
    """
    load environment with given config
    params:
        task: str, name of the environment/task
        robot: str, name of the robot
        config: dict, additional configuration for the environment, including robot, task, etc.
        time_limit: int, maximum time steps for the environment
        reset_wait_step: int, number of steps to wait after reset, using for initialize the scene with no collision
    """
    # load config
    #1.加载任务配置
    import copy
    task_series = find_key_by_value(name2config, task)
    specific_config = copy.deepcopy(config.get(task_series, {}))
    default_config = copy.deepcopy(TASK_CONFIG["default"])
    default_config.update(specific_config)
    # if config is not None and isinstance(config, dict):
    #     default_config.update(config)
    # load and update robot config first and then load robot entity
    #2.加载机器人配置
    robot_config = ROBOT_CONFIG.get(robot, None)
    assert robot_config is not None, f"robot {robot} is not supported"
    robot_config_overide = default_config.get("robot", {})
    robot_config.update(robot_config_overide)
    robot = register.load_robot(robot)(**robot_config)
    #3.加载机器人实体
    task = register.load_task(task)(task, robot, config=default_config, **kwargs)
    # Save the configuration for deterministic evaluation
    # if default_config is not None and "task" in default_config:
    #     # Create a deep copy to avoid modifying the original config
    #     import copy
    #     config_to_save = copy.deepcopy(default_config)
    #     # Save the configuration for later reference
    #     import pickle
    #     import os
    #     # Ensure the directory exists
    #     save_dir = os.path.join(os.getenv("VLABENCH_ROOT"), "configs")
    #     os.makedirs(save_dir, exist_ok=True)
    #     # Save the configuration
    #     with open(os.path.join(save_dir, "default_config.pkl"), "wb") as f:
    #         pickle.dump(config_to_save, f)
    #4.创建环境实例
    env = LM4ManipDMEnv(task=task, time_limit=time_limit, reset_wait_step=reset_wait_step)
    return env