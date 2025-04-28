

from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openvla import OpenVLA
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.evaluation.model.policy.client import RemoteAgentClient
from VLABench.tasks import *
from VLABench.robots import *
import transformers

tasks=[
    "select_fruit_table0", # the pretrain task
    "select_fruit_table1",
    "select_fruit_table2",
    "select_fruit_table3",
    "select_fruit_table4",


    "select_fruit_difficult_table0", # the pretrain task
    "select_fruit_difficult_table1",
    "select_fruit_difficult_table2",
    "select_fruit_difficult_table3",
    "select_fruit_difficult_table4",

    "select_fruit_random_position",     #randomness in [-10,-5] [5,10] cm


    "add_condiment",
    "insert_flower",
    "select_chemistry_tube",

    "select_fruit_difficult",
    "add_condiment_difficult",
    "insert_flower_difficult",
    "select_chemistry_tube_difficult",

]


demo_tasks = ["select_fruit_difficult_table1"]
unseen = False
save_dir = "/mnt/data/310_jiarui/VLABench/logs"

from huggingface_hub import login
from pathlib import Path
import os
os.environ["MUJOCO_GL"] = "egl"
evaluator = Evaluator(
    tasks=demo_tasks,
    n_episodes=50,     #配置评测次数
    max_substeps=10,   
    save_dir=save_dir,
    visulization=True,
    observation_images=["observation.image_1","observation.image_2","observation.image_3","observation.image_4"]  # 可以传入任意的image，但请与训练的时候保持一致。
    #observation_images=["observation.image_1","observation.image_2"]
    #observation_images= ["observation.image_0"]
)
policy = RemoteAgentClient(model="VQ_BET")
result = evaluator.evaluate (policy)

