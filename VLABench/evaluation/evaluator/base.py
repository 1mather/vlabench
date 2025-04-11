import json
import os
import numpy as np
import random
import mediapy
from tqdm import tqdm
from VLABench.envs import load_env
from VLABench.utils.utils import euler_to_quaternion
from scipy.spatial.transform import Rotation as R
import pdb
CAMERA_VIEW_INDEX={
    "select_painting": 1,
    "put_box_on_painting": 1,
    "select_chemistry_tube":2,
    "find_unseen_object":2,
    "texas_holdem": 2,
    "cluster_toy": 2,
    "select_fruit":2
}
def quat2euler(quat, is_degree=False):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles
class Evaluator:
    def __init__(self, 
                 tasks,
                 n_episodes,
                 episode_config=None,
                 max_substeps=10,
                 tolerance=1e-2,
                 metrics=["success_rate"],
                 #metrics=['progress_score'],
                 save_dir=None,
                 visulization=False,
                 **kwargs
                 ):
        """
        Basic evaluator of policy
        params:
            tasks: list of task names to evaluate, e.g. ["task1", "task2"]
            n_episodes: number of episodes to evaluate in each task
            episode_config: dict or path of config file for episode generation
            max_substeps: maximum number of substeps for env.step
            metrics: list of metrics to evaluate
            save_dir: directory to save the evaluation results
            visulization: whether to visualize the evaluation progress as videos
        """
        if isinstance(episode_config, str):
            with open(episode_config, "r") as f:
                self.episode_config = json.load(f)
        else:self.episode_config = episode_config
        if self.episode_config is None:
            print("Load the task episodes by seeds, instead of episodes")
        else:
            assert len(self.episode_config) >= len(n_episodes), "The number of episodes should be less than the number of configurations"
        self.eval_tasks = tasks
        self.n_episodes = n_episodes 
        
        self.max_substeps = max_substeps
        self.tolerance = tolerance
        self.target_metrics = metrics
        
        # log, store and visualization
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.visulization = visulization
        
    def evaluate(self, agent):
        """
        Evaluate the agent on all tasks defined in the evaluator.
        """   
        metrics = {}
        instruction={}
        for task in self.eval_tasks:
            task_infos = []
            instructions=[]
            for i in tqdm(range(self.n_episodes), desc=f"Evaluating {task} of {agent.name}"):
                kwargs = {
                    "unnorm_key": task
                }
                if self.episode_config is None: 
                    info,obs = self.evaluate_single_episode(agent, task, i, None, seed=42+i, **kwargs)
                else: 
                    info,obs= self.evaluate_single_episode(agent, task, i, self.episode_config[i], **kwargs)
                if obs["instruction"] is not None:
                    print(obs["instruction"])
                else:
                    print("there is no instruction")
                task_infos.append(info)
                instructions.append(obs["instruction"])
            metric_score = self.compute_metric(task_infos)       
            metrics[task] = metric_score
            instruction[task]=instructions
            
        if self.save_dir is not None:
            os.makedirs(os.path.join(self.save_dir, agent.name),exist_ok=True)
            with open(os.path.join(self.save_dir, agent.name, "metrics.json"), "w") as f:
                json.dump(metrics, f)
            with open(os.path.join(self.save_dir, agent.name, "instruction.json"), "w") as f:
                json.dump(instruction, f)
        return metrics
        
    def evaluate_single_episode(self, agent, task_name, episode_id, episode_config, seed=42, max_episode_length=200, **kwargs):
        """
        If episode_config is given, the task and scene will load deterministically.
        params:
            agent: policy to evaluate
            task_name: name of the task
            episode_id: id of the episode
            episode_config: configuration of the task
            seed: seed for the random number generator, if episode_config is None
            max_episode_length: maximum length of the episode
        """
        if episode_config is None: # use random seed to ditermine the task
            np.random.seed(seed)
            random.seed(seed)
        env = load_env(task_name, config=episode_config)
        env.reset()
        #pdb.set_trace()
        success = False
        info = {}
        frames_to_save = []
        view_of_model=[]
        for i in range(max_episode_length):
            observation = env.get_observation()
            observation["instruction"] = env.task.get_instruction()
            if self.save_dir is not None and self.visulization:
                frames_to_save.append(observation["rgb"])
                cam_index = CAMERA_VIEW_INDEX.get(task_name, 0)

                view_of_model.append(observation["rgb"][cam_index])
            if agent.control_mode == "ee":
                #pdb.set_trace()
                #但是注意这里输出的应该是delta_action, 具体实现在VLABench/evaluation/model/policy/Openvla.py
            
                """
                这里可以添加多个接口用于测评不同的模型
                """
                from tutorials.test_client import send_test_request
                from VLABench.evaluation.model.policy.client import RemoteAgentClient
                if isinstance(agent, RemoteAgentClient):
                    ee_state = observation["ee_state"]
                    ee_pos, ee_quat, gripper = ee_state[:3], ee_state[3:7], np.array([ee_state[7]])
                    ee_euler = quat2euler(ee_quat)
                    ee_pos -= np.array([0, -0.4, 0.78])
                    print(ee_pos,ee_euler,gripper)
                    ee_state = np.concatenate([ee_pos, ee_euler, gripper], axis=0)
                    try:
                        pos, euler, gripper_state, view_index = send_test_request(observation["rgb"][cam_index],ee_state)
                    except Exception as e:
                        continue
                else:
                    pos, euler, gripper_state, view_index = agent.predict(observation, **kwargs)
                quat = euler_to_quaternion(*euler)
                action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=quat)[:7]#delta关节角度
                action = np.concatenate([action, gripper_state])
            elif agent.control_mode == "joint":
                qpos, gripper_state = agent.predict(observation, **kwargs)
                action = np.concatenate([qpos, gripper_state])
            else:
                raise NotImplementedError(f"Control mode {agent.control_mode} is not implemented")    
            


            for _ in range(self.max_substeps):
                timestep = env.step(action)
                if timestep.last():
                    success=True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos-np.array(action)[:7]) < self.tolerance \
                    and np.min(current_qpos - np.array(action)[:7]) > -self.tolerance:
                    break
            if success:
                break
            #这里进行的操作是：


        info["task"] = task_name
        info["success"] = success
        info["consumed_step"] = i
        #info["intention_score"] = env.get_intention_score()
        info["progress_score"] = env.get_task_progress()
        
        env.close()
        if self.save_dir is not None and self.visulization:
            os.makedirs(os.path.join(self.save_dir, agent.name, task_name), exist_ok=True)
            episode_id=str(episode_id)+"view"+str(view_index)
            if success:
                self.save_video(frames_to_save, os.path.join(self.save_dir, agent.name, task_name, f"{episode_id}_success.mp4"))
            else:
                self.save_video(frames_to_save, os.path.join(self.save_dir, agent.name, task_name, f"{episode_id}_fail.mp4"))
            mediapy.write_video(os.path.join(self.save_dir, agent.name, task_name, f"{episode_id}_view_of_model.mp4"),view_of_model, fps=10) 
        return info,observation
        
    def compute_metric(self, infos):
        """
        Compute the metric scores for the evaluation
        param:
            infos: list of episode information
        """
        metric = {}
        for key in self.target_metrics:
            if key == "success_rate": # compute the success rate
                success = [info["success"] for info in infos]
                sucess_rate = np.mean(success)
                metric["success_rate"] = sucess_rate
            elif key == "intention_score":
                intention_score = [info["intention_score"] for info in infos]
                avg_intention_score = np.mean(intention_score)
                metric["intention_score"] = avg_intention_score
            elif key == "progress_score":
                progress_score = [info["progress_score"]for info in infos]
                #avg_progress_score = np.mean(progress_score)
                metric["progress_score"] = progress_score
            else:
                raise NotImplementedError(f"Metric {key} is not implemented")
        return metric
    
    def save_video(self, frames, save_dir):
        frames_to_save = [] 
        for frame in frames:
            frames_to_save.append(np.vstack([np.hstack(frame[:2]), np.hstack(frame[2:4])]))
        mediapy.write_video(save_dir, 
                            frames_to_save, fps=10) 