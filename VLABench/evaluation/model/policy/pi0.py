from peft import PeftModel, PeftConfig
import torch
import os
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from PIL import Image
from VLABench.evaluation.model.policy.base import Policy 
from VLABench.utils.utils import quaternion_to_euler
from pathlib import Path
import json
import pdb
DEBUG=False
CAMERA_VIEW_INDEX={
    "select_painting": 1,
    "put_box_on_painting": 1,
    "select_chemistry_tube":2,
    "find_unseen_object":2,
    "texas_holdem": 2,
    "cluster_toy": 2,
    "select_fruit":0
}

def copy_file_content(content_file, target_file):
    with open(content_file, "r") as f:
        content = f.read()
    with open(target_file, "w") as f:
        f.write(content)

class Pi0(Policy):
    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    def __init__(self, 
                 paligamma_path,  # PaliGemma 模型路径
                 gemma_expert_path,  # Gemma Expert 模型路径
                 device="cuda",
                 **kwargs):
        
        self.device = torch.device(device)
        
        # 初始化 PaliGemma
        self.paligamma = AutoModelForVision2Seq.from_pretrained(
            paligamma_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        
        # 初始化 Gemma Expert
        self.gemma_experts = []
        for i in range(10):  # 10个 Gemma Expert
            expert = AutoModelForVision2Seq.from_pretrained(
                gemma_expert_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(self.device)
            self.gemma_experts.append(expert)
            
        self.processor = AutoProcessor.from_pretrained(paligamma_path, trust_remote_code=True)
        super().__init__(None)  # 不使用基类的model
        
    def process_observation(self, obs, unnorm_key):
        cam_index = CAMERA_VIEW_INDEX.get(unnorm_key, 0)
        instruction = obs["instruction"]
        rgb = obs["rgb"][cam_index]
        
        # 处理图像
        image = Image.fromarray(rgb).convert("RGB")
        processed_image = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # 处理指令
        text_inputs = self.processor(
            text=instruction,
            return_tensors="pt"
        ).to(self.device)
        
        return {
            "image": processed_image,
            "text": text_inputs,
            "robot_state": torch.tensor(obs["ee_state"]).to(self.device)
        }
    
    def predict(self, obs, unnorm_key=None):
        inputs = self.process_observation(obs, unnorm_key)
        
        # PaliGemma 生成 kv cache
        with torch.no_grad():
            kv_cache = self.paligamma(
                inputs["image"],
                inputs["text"]
            )
        
        # 生成噪声
        noise = torch.randn(1, 10, device=self.device)  # 假设噪声维度为10
        
        # Gemma Expert 生成动作
        actions = []
        for expert in self.gemma_experts:
            with torch.no_grad():
                action = expert(
                    kv_cache=kv_cache,
                    robot_state=inputs["robot_state"],
                    noise=noise
                )
            actions.append(action)
        
        # 合并所有专家的动作
        final_action = torch.mean(torch.stack(actions), dim=0)
        
        # 转换为目标位置、欧拉角和夹持器状态
        current_ee_state = obs["ee_state"]
        if len(current_ee_state) == 8:
            pos, quat = current_ee_state[:3], current_ee_state[3:7]
            euler = quaternion_to_euler(quat)
        elif len(current_ee_state) == 7:
            pos, euler = current_ee_state[:3], current_ee_state[3:6]
            
        delta_action = final_action.cpu().numpy()
        target_pos = np.array(pos) + delta_action[:3]
        target_euler = euler + delta_action[3:6]
        gripper_open = delta_action[-1]
        gripper_state = np.ones(2)*0.04 if gripper_open >= 0.1 else np.zeros(2)
        
        view_index = CAMERA_VIEW_INDEX.get(unnorm_key, 0)
        return target_pos, target_euler, gripper_state, view_index
    
    @property
    def name(self):
        return "LeRobot-Pi0"