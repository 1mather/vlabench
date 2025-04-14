

from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openvla import OpenVLA
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.evaluation.model.policy.client import RemoteAgentClient
from VLABench.tasks import *
from VLABench.robots import *
import transformers
demo_tasks = ["select_fruit"]
unseen = False
save_dir = "/mnt/data/310_jiarui/VLABench/logs"
#/home/tyj/Documents/310_jiarui/openvla/log/train_log/openvla-7b+h5py_dataset+b1+lr-0.0005+lora-r8+dropout-0.0
# model_ckpt = "/mnt/data/310_jiarui/VLABench/model_parameter/base/openvla-7b+vlabench_dataset+b80+lr-0.0005+lora-r16+dropout-0.0--time-20250408-13"
# lora_ckpt ="/mnt/data/310_jiarui/VLABench/model_parameter/adapter/openvla-7b+vlabench_dataset+b80+lr-0.0005+lora-r16+dropout-0.0--time-20250408-13"


from huggingface_hub import login
from pathlib import Path
import os
os.environ["MUJOCO_GL"] = "egl"
evaluator = Evaluator(
    tasks=demo_tasks,
    n_episodes=500,     #配置评测次数
    max_substeps=10,   
    save_dir=save_dir,
    visulization=True,
    observation_images=["observation.image_0","observation.image_1"]  # 可以传入任意的image，但请与训练的时候保持一致。
)

policy = RemoteAgentClient(model="VQ_BET")
result = evaluator.evaluate (policy)

