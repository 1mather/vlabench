import numpy as np
from scipy.spatial.transform import Rotation
import json

def parse_xyaxes(xyaxes_str):
    """解析 xyaxes 字符串为 x 轴和 y 轴向量"""
    numbers = [float(x) for x in xyaxes_str.split()]
    x_axis = np.array(numbers[0:3])
    y_axis = np.array(numbers[3:6])
    return x_axis, y_axis

def axes_to_quaternion(x_axis, y_axis):
    """将 x 轴和 y 轴转换为四元数"""
    # 归一化 x 轴
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 计算 z 轴（叉乘）
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # 重新计算 y 轴以确保正交性
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 构建旋转矩阵
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    
    # 转换为四元数
    r = Rotation.from_matrix(rotation_matrix)
    quat = r.as_quat()  # 返回 [x, y, z, w] 格式的四元数
    
    return quat.tolist()

# 相机参数数据
cameras_data = {
    "right0": {
        "position": [0.775, -0.856, 1.209],
        "xyaxes": "0.733 0.681 0.000 -0.134 0.144 0.981"
    },
    "right1": {
        "position": [1, 1, 1.3],
        "xyaxes": "-0.733 0.681 0.000 -0.134 -0.144 0.981"
    },
    "left0": {
        "position": [-0.775, -0.856, 1.209],
        "xyaxes": "0.733 -0.681 0.000 0.134 0.144 0.981"
    },
    "left1": {
        "position": [-1, 1, 1.3],
        "xyaxes": "-0.733 -0.681 0.000 0.134 -0.144 0.981"
    },
    "forward": {
        "position": [-0.016, 1.223, 1.644],
        "xyaxes": "-1.000 -0.015 -0.000 0.008 -0.551 0.834"
    }
}

# 机器人基座偏移量
ROBOT_OFFSET = np.array([0, -0.4, 0.78])

# 创建输出字典
camera_params = {}

# 处理每个相机的参数
for camera_name, data in cameras_data.items():
    # 将位置从世界坐标系转换到机器人基座坐标系
    world_pos = np.array(data["position"])
    position = (world_pos - ROBOT_OFFSET).tolist()
    
    x_axis, y_axis = parse_xyaxes(data["xyaxes"])
    quaternion = axes_to_quaternion(x_axis, y_axis)
    
    camera_params[camera_name] = {
        "position": position,
        "quaternion": quaternion
    }
    
    # 打印转换结果
    print(f"\n{camera_name}:")
    print(f"World Position: {data['position']}")
    print(f"Robot Base Position: {position}")
    print(f"Quaternion [x, y, z, w]: {quaternion}")

# 保存为 JSON 文件
with open('camera_params.json', 'w') as f:
    json.dump(camera_params, f, indent=4)

print("\nCamera parameters have been saved to camera_params.json")