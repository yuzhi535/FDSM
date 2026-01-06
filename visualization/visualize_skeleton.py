import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random
import csv

# NTU RGB+D 骨架连接关系（25个关节点）
NTU_SKELETON_BONES = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
]

# 关节点名称
JOINT_NAMES = [
    'base of spine', 'middle of spine', 'neck', 'head',
    'left shoulder', 'left elbow', 'left wrist', 'left hand',
    'right shoulder', 'right elbow', 'right wrist', 'right hand',
    'left hip', 'left knee', 'left ankle', 'left foot',
    'right hip', 'right knee', 'right ankle', 'right foot',
    'spine', 'tip of left hand', 'left thumb',
    'tip of right hand', 'right thumb'
]


def load_action_labels(csv_path='./data/class_lists/ntu120.csv'):
    """加载动作标签"""
    labels = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[int(row['idx'])] = row['label']
    except Exception as e:
        print(f"加载标签文件失败: {e}")
    return labels


def extract_action_id(filename):
    """从文件名提取动作ID"""
    try:
        filename_without_ext = filename.replace('.skeleton', '')
        action_idx = filename_without_ext.find('A')
        if action_idx != -1:
            action_id = int(filename_without_ext[action_idx+1:action_idx+5])
            return action_id
    except Exception as e:
        pass
    return None


def read_skeleton_file(file_path):
    """读取NTU RGB+D的.skeleton文件"""
    with open(file_path, 'r') as f:
        frame_count = int(f.readline())
        frames_data = []
        
        for frame in range(frame_count):
            body_count = int(f.readline())
            frame_bodies = []
            
            for body in range(body_count):
                body_info = f.readline().split()
                joint_count = int(f.readline())
                
                joints = []
                for joint in range(joint_count):
                    joint_info = f.readline().split()
                    # x, y, z 坐标
                    x, y, z = float(joint_info[0]), float(joint_info[1]), float(joint_info[2])
                    joints.append([x, y, z])
                
                frame_bodies.append(np.array(joints))
            
            frames_data.append(frame_bodies)
    
    return frames_data


def plot_skeleton_frame(ax, skeleton, title="", coord_range=None):
    """在3D坐标系中绘制单帧骨架"""
    if len(skeleton) == 0:
        return
    
    # 取第一个人的骨架
    joints = skeleton[0] if len(skeleton) > 0 else np.zeros((25, 3))
    
    # NTU RGB+D坐标系正确映射到matplotlib 3D显示
    # 原始: X(左右), Y(高度), Z(深度)
    # matplotlib 3D: x轴(左右), y轴(深度), z轴(高度-垂直)
    x = joints[:, 0]  # 左右 -> x轴
    y = joints[:, 2]  # 深度 -> y轴  
    z = joints[:, 1]  # 高度 -> z轴（垂直显示）
    
    # 绘制关节点
    ax.scatter(x, y, z, c='red', marker='o', s=50, alpha=0.8)
    
    # 绘制骨骼连接
    for bone in NTU_SKELETON_BONES:
        start_idx, end_idx = bone[0] - 1, bone[1] - 1  # 索引从0开始
        if start_idx < len(joints) and end_idx < len(joints):
            ax.plot([x[start_idx], x[end_idx]],
                   [y[start_idx], y[end_idx]],
                   [z[start_idx], z[end_idx]],
                   'b-', linewidth=2, alpha=0.6)
    
    # 隐藏坐标轴
    ax.set_axis_off()
    
    ax.set_title(title, fontsize=10)
    
    # 使用传入的坐标范围或自动计算
    if coord_range is not None:
        x_range, y_range, z_range = coord_range
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
    else:
        # 自动根据当前帧数据设置范围
        if len(joints) > 0:
            padding = 0.2
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            z_min, z_max = z.min(), z.max()
            
            ax.set_xlim([x_min - padding, x_max + padding])
            ax.set_ylim([y_min - padding, y_max + padding])
            ax.set_zlim([z_min - padding, z_max + padding])


def visualize_skeleton_sequence(skeleton_file, output_path='skeleton_preview.png', num_frames=6, labels=None):
    """可视化骨架序列的多个帧"""
    print(f"读取骨架文件: {skeleton_file}")
    frames_data = read_skeleton_file(skeleton_file)
    
    # 提取动作标签
    filename = os.path.basename(skeleton_file)
    action_id = extract_action_id(filename)
    action_label = ""
    if action_id and labels and action_id in labels:
        action_label = labels[action_id]
        print(f"动作ID: {action_id}, 标签: {action_label}")
    
    total_frames = len(frames_data)
    print(f"总帧数: {total_frames}")
    
    if total_frames == 0:
        print("错误：文件中没有帧数据")
        return
    
    # 选择要显示的帧索引
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames // num_frames
        frame_indices = [i * step for i in range(num_frames)]
    
    # 计算整个序列的坐标范围（用于统一所有子图的显示范围）
    all_joints = []
    for frame_idx in frame_indices:
        if len(frames_data[frame_idx]) > 0:
            all_joints.append(frames_data[frame_idx][0])
    
    if len(all_joints) > 0:
        all_joints = np.vstack(all_joints)
        padding = 0.3
        # 坐标映射到matplotlib 3D: x(左右), y(深度), z(高度)
        x_min, x_max = all_joints[:, 0].min() - padding, all_joints[:, 0].max() + padding
        y_min, y_max = all_joints[:, 2].min() - padding, all_joints[:, 2].max() + padding  # 深度
        z_min, z_max = all_joints[:, 1].min() - padding, all_joints[:, 1].max() + padding  # 高度
        coord_range = ([x_min, x_max], [y_min, y_max], [z_min, z_max])
    else:
        coord_range = None
    
    # 创建子图
    rows = 2
    cols = 3
    fig = plt.figure(figsize=(16, 10))
    
    # 添加标签作为标题
    if action_label:
        fig.suptitle(action_label, fontsize=13, fontweight='bold', wrap=True)
    else:
        fig.suptitle(f'Action ID: {action_id}', fontsize=13, fontweight='bold')
    
    for idx, frame_idx in enumerate(frame_indices[:num_frames]):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        plot_skeleton_frame(ax, frames_data[frame_idx], 
                          title=f'Frame {frame_idx+1}/{total_frames}',
                          coord_range=coord_range)
        ax.view_init(elev=15, azim=-75)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 骨架预览已保存至: {output_path}")
    plt.close()


def visualize_random_samples(dataset_path, num_samples=3, output_dir='skeleton_previews'):
    """随机可视化数据集中的几个样本"""
    # 加载标签
    labels = load_action_labels()
    
    # 获取所有.skeleton文件
    skeleton_files = [f for f in os.listdir(dataset_path) if f.endswith('.skeleton')]
    
    if len(skeleton_files) == 0:
        print(f"在 {dataset_path} 中没有找到.skeleton文件")
        return
    
    print(f"找到 {len(skeleton_files)} 个骨架文件")
    
    # 随机选择样本
    selected_files = random.sample(skeleton_files, min(num_samples, len(skeleton_files)))
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    for i, filename in enumerate(selected_files):
        file_path = os.path.join(dataset_path, filename)
        output_path = os.path.join(output_dir, f'sample_{i+1}_{filename[:-9]}.png')
        
        print(f"\n处理样本 {i+1}/{num_samples}: {filename}")
        try:
            visualize_skeleton_sequence(file_path, output_path, num_frames=6, labels=labels)
        except Exception as e:
            print(f"错误: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化NTU RGB+D骨架数据')
    parser.add_argument('--file', type=str, default=None, 
                        help='单个.skeleton文件路径')
    parser.add_argument('--dataset', type=str, 
                        default='./datasets/nturgbd_raw/',
                        help='数据集目录路径')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='随机可视化的样本数量')
    parser.add_argument('--output', type=str, default='skeleton_preview.png',
                        help='输出图片路径')
    
    args = parser.parse_args()
    
    if args.file:
        # 可视化单个文件
        labels = load_action_labels()
        visualize_skeleton_sequence(args.file, args.output, labels=labels)
    else:
        # 随机可视化多个样本
        visualize_random_samples(args.dataset, args.num_samples)
