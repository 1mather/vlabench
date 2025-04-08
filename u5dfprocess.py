import h5py
import numpy as np
import cv2
import glob
import os
from PIL import Image
import pdb
def read_rgb_from_u5df(file_path, view_index=2, output_path=None, fps=10, rewrite=False):
    
    try:
        with h5py.File(file_path, "r") as f:
            timestamp_key = list(f["data"].keys())[0]
            rgb_data = f["data"][timestamp_key]["observation"]["rgb"][()]
            frames = rgb_data[:-4] # 去除掉倒数两四帧
        
            data_obs = {}
            for key in f["data"][timestamp_key]["observation"].keys():
                data_obs[key] = f["data"][timestamp_key]["observation"][key][:-4]  # 去掉倒数四帧
            data_traj = {}
            data_traj["trajectory"] = f["data"][timestamp_key]["trajectory"][:-4]
            data_inst={}
            data_inst["instruction"] = f["data"][timestamp_key]["instruction"][()]
        # 如果没有指定输出路径，在原文件名后添加 _processed
        if rewrite:          
            base_name = os.path.splitext(os.path.basename(file_path))[0]  # 只获取文件名部分
            output_folder = os.path.join(os.path.dirname(file_path), "rewrite")
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"processed_{base_name}.hdf5")
         # 先创建文件和组结构
        with h5py.File(output_path, "w") as f:
            # 创建必要的组结构
            data_group = f.create_group("data")
            timestamp_group = data_group.create_group(timestamp_key)
            obs_group = timestamp_group.create_group("observation")
            # 然后写入数据
            for key, value in data_obs.items():
                obs_group.create_dataset(key, data=value,compression='gzip', compression_opts=9)
            timestamp_group.create_dataset("trajectory", data=data_traj["trajectory"],compression='gzip', compression_opts=9)
            timestamp_group.create_dataset("instruction", data=data_inst["instruction"],compression='gzip', compression_opts=9)
        print(f"✅ Saved processed u5df: {output_path}")
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
    
    # 直接使用输入文件的base_name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # 创建输出目录
    output_folder = os.path.join(os.path.dirname(file_path), "video")
    os.makedirs(output_folder, exist_ok=True)
    # 使用base_name作为输出文件名
    output_path = os.path.join(output_folder, f"{base_name}.mp4")
    #height, width = frames[0].shape
    height=frames[0].shape[1]*3
    width=frames[0].shape[2]*2
    print(f"height: {height}, width: {width}")
    # 添加错误检查
    try:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        if not out.isOpened():
            raise Exception(f"Failed to create video writer for path: {output_path}")
        
        for frame in frames:
            frame=np.vstack((np.hstack((frame[0],frame[1])),np.hstack((frame[2],frame[3])),np.hstack((frame[4],frame[5]))))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"✅ Saved video: {output_path}")
    except Exception as e:
        print(f"Error saving video: {str(e)}")



def print_action_sequence(file_path, print_last_only=False):
    try:
        with h5py.File(file_path, "r") as f:
            timestamp_key = list(f["data"].keys())[0]
            actions = f["data"][timestamp_key]["trajectory"]  # (T, 8)

            if actions.shape[1] == 8:
                print(f"\n📁 File: {file_path}")
                print(f"🟢 Action sequence shape: {actions.shape}")

                for i, act in enumerate(actions):
                    last_element = act[-1]
                    if print_last_only:
                        print(f"[{i:03}] Gripper State: {last_element:.4f}")
                    else:
                        print(f"[{i:03}] Action: {act} | Gripper: {last_element:.4f}")
            else:
                print(f"⚠️ Skipping {file_path}: Unexpected action shape {actions.shape}")

    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")

import h5py
import numpy as np
from PIL import Image
import os

def clean_the_empty(file_path, view_index=2, save_dir="/mnt/data/310_jiarui/VLABench/media_dataset/select_fruit/last1_frame"):
    print(f"\n🔍 Processing: {file_path}")

    try:
        with h5py.File(file_path, "r") as f:
            # Get the timestamp key
            timestamp_keys = list(f["data"].keys())
            if not timestamp_keys:
                raise ValueError("No timestamp keys found in file")
            timestamp_key = timestamp_keys[0]

            # Read RGB frames
            rgb_dataset = f["data"][timestamp_key]["observation"]["rgb"]
            rgb_data = rgb_dataset[:]  # force read
            if rgb_data.shape[0] == 0:
                raise ValueError("RGB data is empty")
            if view_index >= rgb_data.shape[1]:
                raise ValueError(f"View index {view_index} out of range")
            print(f"RGB shape: {rgb_data.shape}")
            last_frame = rgb_data[-1, view_index]  # (H, W, 3)

            # Read trajectory
            traj_dataset = f["data"][timestamp_key]["trajectory"]
            traj_data = traj_dataset[:]  # force read
            if traj_data.shape[0] == 0:
                raise ValueError("Trajectory is empty")
            print(f"Trajectory shape: {traj_data.shape}")
            last_action = traj_data[-1]

        # Save the image
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        img_path = os.path.join(save_dir, f"{file_name}_view{view_index}_last.png")
        Image.fromarray(last_frame).save(img_path)

        print(f"\n✅ Saved last frame: {img_path}")
        print(f"🟢 Last action: {last_action}")
        print(f"🤖 Gripper state (last element): {last_action[-1]:.4f}")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted corrupted file: {file_path}")
        except Exception as delete_err:
            print(f"⚠️ Failed to delete file: {delete_err}")

# 🔁 Batch process
#for u5df_file in glob.glob("/mnt/data/310_jiarui/VLABench/media/deterministic_plate_random/select_fruit/*.hdf5"):
for u5df_file in glob.glob("/mnt/data/310_jiarui/VLABench/media/deterministic_v1/select_fruit/*.hdf5"):
    read_rgb_from_u5df(u5df_file, view_index=2,output_path="/mnt/data/310_jiarui/VLABench/media/deterministic_v1/select_fruit/select_fruit.mp4",rewrite=True)
    # save_last_frame(u5df_file)
    # print_action_sequence(u5df_file,True)
    # clean_the_empty()

# for u5df_file in glob.glob("/mnt/data/310_jiarui/VLABench/media/deterministic_v1/select_fruit/rewrite/*.hdf5"):
#     #read_rgb_from_u5df(u5df_file, view_index=2,output_path="/mnt/data/310_jiarui/VLABench/media/deterministic_plate_random/select_fruit/rewrite/select_fruit.mp4",rewrite=True)
#     #print_action_sequence(u5df_file,True)
#     clean_the_empty(u5df_file,save_dir="/mnt/data/310_jiarui/VLABench/media/deterministic_v1/select_fruit/rewrite/cleancheck")