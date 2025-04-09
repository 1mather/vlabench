from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
import argparse
from pathlib import Path
import shutil

def convert_feature_names(args):
    # 确保输入和输出路径正确
    # import pdb; pdb.set_trace()
    # input_root = Path(args.root_dir).expanduser()
    if args.output_dir:
        output_root = Path(args.output_dir).expanduser()
    else:
        output_root = input_root
    
    # 使用本地数据集
    print(f"Loading dataset from {input_root / args.input_dataset_name}")
    original_dataset = LeRobotDataset(
        args.input_dataset_name, 
        root=input_root,
        local_files_only=True
    )
    
    print(f"Original dataset loaded with {len(original_dataset)} frames")
    
    # 检查输出目录是否已存在，如果存在则先删除
    output_path = output_root / args.output_dataset_name
    if output_path.exists():
        print(f"Output directory {output_path} already exists. Removing it first.")
        shutil.rmtree(output_path)
    
    # 创建新的数据集
    features = {}
    
    # 获取原始数据集的特征
    original_features = original_dataset.meta.features
    print(f"Original features: {original_features}")
    
    # 转换特征名称
    for key, value in original_features.items():
        if not key.startswith("observation.") and key != "action" and not key.startswith("next.") and not key.startswith("episode_") and key != "timestamp" and key != "index":
            # 添加 observation. 前缀
            new_key = f"observation.{key}"
            print(f"Converting {key} to {new_key}")
            features[new_key] = value
        else:
            # 保持原样
            features[key] = value
    
    print(f"New features: {features}")
    
    # 创建新的数据集
    dataset = LeRobotDataset.create(
        repo_id=args.output_dataset_name,
          # 使用输出根目录
        robot_type=original_dataset.meta.info.get("robot_type", "franka"),
        fps=original_dataset.meta.info.get("fps", 10),
        features=features,
        image_writer_processes=5,
        image_writer_threads=10
    )
    
    # 特殊字段列表，这些字段不应添加observation.前缀
    special_fields = ["action", "timestamp", "index", "episode_index", "frame_index", "next.done", "next.reward", "next.success"]
    
    # 遍历原始数据集的每个episode
    for episode_idx in range(len(original_dataset.episode_data_index["from"])):
        episode_start = original_dataset.episode_data_index["from"][episode_idx].item()
        episode_end = original_dataset.episode_data_index["to"][episode_idx].item()
        
        print(f"Processing episode {episode_idx} with frames {episode_start} to {episode_end}")
        
        # 获取该episode的所有帧
        for idx in range(episode_start, episode_end + 1):
            frame_data = original_dataset[idx]
            
            # 转换特征名称
            new_frame_data = {}
            for key, value in frame_data.items():
                if key in special_fields or key.startswith("next.") or key.startswith("episode_") or key.startswith("observation."):
                    # 保持原样
                    new_frame_data[key] = value
                else:
                    # 添加 observation. 前缀
                    new_key = f"observation.{key}"
                    new_frame_data[new_key] = value
            
            # 添加帧到新数据集
            dataset.add_frame(new_frame_data)
        
        # 保存当前episode
        task_name = f"episode_{episode_idx}"
        if hasattr(original_dataset.meta, "episodes") and len(original_dataset.meta.episodes) > episode_idx and "task" in original_dataset.meta.episodes[episode_idx]:
            task_name = original_dataset.meta.episodes[episode_idx]["task"]
        
        dataset.save_episode(task=task_name)
    
    # 整合数据集并计算统计信息
    dataset.consolidate(run_compute_stats=True)
    
    print(f"Dataset converted successfully! New dataset: {args.output_dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert feature names in a LeRobot dataset")
    parser.add_argument("--input-dataset-name", type=str, required=True, help="Name of the input dataset")
    parser.add_argument("--output-dataset-name", type=str, required=True, help="Name of the output dataset")
    parser.add_argument("--root-dir", type=str, default="~/.cache/huggingface", help="Root directory of the dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for the new dataset")
    
    args = parser.parse_args()
    convert_feature_names(args)