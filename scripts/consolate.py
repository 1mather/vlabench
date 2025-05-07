#!/usr/bin/env python3
# direct_consolidate.py - 直接整合未完成的LeRobot数据集

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import argparse
import os
import glob

def direct_consolidate_dataset(dataset_path,episodes=None):
    """
    直接整合未完成的数据集，不依赖于元数据文件
    
    参数:
        dataset_path: 数据集路径
    """
    # 从路径中提取数据集名称
    dataset_name = "add_condiment_4_800"
    print(f"正在尝试直接整合数据集: {dataset_name} 从路径: {dataset_path}")

    try:
        # 尝试直接加载数据集
        print("尝试直接加载数据集...")
        if episodes is None:
            episodes=None
        else:
            episodes=[i for i in range(episodes)]
        dataset = LeRobotDataset(repo_id=dataset_name,tolerance_s=0.2,root=dataset_path,local_files_only=True,episodes=episodes)
        
        print("开始整合数据...")
        dataset.consolidate(run_compute_stats=True)
        print("数据整合完成！")
        return
        
    except Exception as e:
        print(f"直接加载失败: {str(e)}")
    
    try:
        # 第二种方法：尝试从原始参数重新创建数据集并链接到现有数据
        print("尝试从原始参数重新创建数据集...")
        
        # 使用与convert_to_lerobot.py中相同的参数

        dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            root=dataset_path,
            robot_type="franka",
            fps=10,
            features={
                "observation.image_0": {
                    "dtype": "image", 
                    "shape": (480, 480, 3),
                    "names": ["height", "width", "channels"]
                },
                "observation.image_1": {
                    "dtype": "image", 
                    "shape": (480, 480, 3),
                    "names": ["height", "width", "channels"]
                },
                "observation.image_2": {
                    "dtype": "image", 
                    "shape": (480, 480, 3),
                    "names": ["height", "width", "channels"]
                },
                "observation.image_3": {
                    "dtype": "image", 
                    "shape": (480, 480, 3),
                    "names": ["height", "width", "channels"]
                },
                "observation.image_4": {
                    "dtype": "image", 
                    "shape": (480, 480, 3),
                    "names": ["height", "width", "channels"]
                },
                "observation.image_wrist": {
                    "dtype": "image", 
                    "shape": (480, 480, 3),
                    "names": ["height", "width", "channels"]
                },
                "observation.state": {
                    "dtype": "float", 
                    "shape": (7,),
                    "names": ["state"]
                },
                "action": {
                    "dtype": "float", 
                    "shape": (7,),
                    "names": ["actions"]
                },
            },
            image_writer_processes=5,
            image_writer_threads=10,

        )
        
        print("开始整合数据...")
        dataset.consolidate(run_compute_stats=True)
        print("数据整合完成！")
        
    except Exception as e:
        print(f"重新创建数据集失败: {str(e)}")
        print("\n您可能需要检查LeRobotDataset库的文档，或者联系开发人员了解如何恢复未完成的数据集。")
        print("数据集路径中可能包含部分处理的数据，如果需要重新开始，您可能需要先备份这些数据。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="直接整合未完成的LeRobot数据集")
    parser.add_argument("--dataset-path", type=str, 
                        default=None,
                        help="数据集路径")
    parser.add_argument("--episodes", type=int,
                        default=None,
                        help="要整合的episodes")
    args = parser.parse_args()
    direct_consolidate_dataset(args.dataset_path,args.episodes)
"""
python consolate.py --dataset-path /home/tyj/.cache/huggingface/lerobot/add_condiment_160_diff
python consolate.py --dataset-path /home/tyj/.cache/huggingface/lerobot/select_chemistry_tube_1_200_diff


mini
python consolate.py --dataset-path /home/tyj/.cache/huggingface/lerobot/add_condiment_160_diff --episodes 100

python consolate.py --dataset-path /home/tyj/.cache/huggingface/lerobot/select_fruit_1_5_diff --episodes 5

python consolate.py --dataset-path /home/tyj/.cache/huggingface/lerobot/select_chemistry_tube_1_50_diff --episodes 50

python consolate.py --dataset-path /home/tyj/.cache/huggingface/lerobot/add_condiment_1_50_diff --episodes 50
"""