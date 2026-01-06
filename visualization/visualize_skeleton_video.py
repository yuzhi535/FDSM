import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
import random
import csv
from matplotlib.animation import FuncAnimation, PillowWriter

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
        print(f"已加载 {len(labels)} 个动作标签")
    except Exception as e:
        print(f"加载标签文件失败: {e}")
    return labels


def extract_action_id(filename):
    """从文件名提取动作ID
    NTU RGB+D文件名格式: SsssccPpprrrAaaa.skeleton
    aaa是动作ID"""
    try:
        # 移除.skeleton扩展名
        filename_without_ext = filename.replace('.skeleton', '')
        # 文件名格式: S018C001P008R001A061
        # 提取A后面的4位数字
        action_idx = filename_without_ext.find('A')
        if action_idx != -1:
            action_id = int(filename_without_ext[action_idx+1:action_idx+5])
            return action_id
    except Exception as e:
        print(f"提取动作ID失败: {e}")
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
                f.readline().split()
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


def create_skeleton_video(skeleton_file, output_path='skeleton_video.gif', fps=10, labels=None):
    """创建骨架动作视频动画"""
    print(f"读取骨架文件: {skeleton_file}")
    frames_data = read_skeleton_file(skeleton_file)
    
    # 提取动作ID和标签
    filename = os.path.basename(skeleton_file)
    action_id = extract_action_id(filename)
    action_label = ""
    if action_id and labels and action_id in labels:
        action_label = labels[action_id]
        print(f"动作ID: {action_id}, 标签: {action_label}")
    else:
        print(f"动作ID: {action_id}, 未找到对应标签")
    
    total_frames = len(frames_data)
    print(f"总帧数: {total_frames}")
    
    if total_frames == 0:
        print("错误：文件中没有帧数据")
        return
    
    # 计算整个序列的坐标范围
    all_joints = []
    for frame_bodies in frames_data:
        if len(frame_bodies) > 0:
            all_joints.append(frame_bodies[0])
    
    all_joints = np.vstack(all_joints)
    padding = 0.3
    # 坐标映射到matplotlib 3D: x(左右), y(深度), z(高度)
    x_min, x_max = all_joints[:, 0].min() - padding, all_joints[:, 0].max() + padding
    y_min, y_max = all_joints[:, 2].min() - padding, all_joints[:, 2].max() + padding  # 深度
    z_min, z_max = all_joints[:, 1].min() - padding, all_joints[:, 1].max() + padding  # 高度
    
    print(f"坐标范围 - X(左右): [{x_min:.2f}, {x_max:.2f}], Y(深度): [{y_min:.2f}, {y_max:.2f}], Z(高度): [{z_min:.2f}, {z_max:.2f}]")
    
    # 创建图表
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 初始化散点图和线条
    scatter = ax.scatter([], [], [], c='red', marker='o', s=50, alpha=0.8)
    lines = [ax.plot([], [], [], 'b-', linewidth=2, alpha=0.6)[0] for _ in NTU_SKELETON_BONES]
    
    # 设置坐标轴
    ax.set_xlabel('X (Left-Right)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    # 调整视角：从前方稍微侧面观察
    ax.view_init(elev=15, azim=-75)
    
    # 标题和进度条
    title_text = ax.set_title('')
    
    # 添加标签文本（显示在图表之外）
    if action_label:
        fig.text(0.5, 0.98, f'{action_label}', 
                 ha='center', va='top', fontsize=12, wrap=True, fontweight='bold')
    else:
        fig.text(0.5, 0.98, f'Action ID: {action_id}', 
                 ha='center', va='top', fontsize=12, wrap=True, fontweight='bold')
    
    def init():
        scatter.set_offsets(np.c_[[], []])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return [scatter] + lines + [title_text]
    
    def update(frame_idx):
        """更新每一帧"""
        if len(frames_data[frame_idx]) == 0:
            return [scatter] + lines + [title_text]
        
        joints = frames_data[frame_idx][0]  # 取第一个人
        
        # NTU RGB+D坐标系正确映射到matplotlib 3D显示
        # 原始: X(左右), Y(高度), Z(深度)
        # matplotlib 3D: x轴(左右), y轴(深度), z轴(高度-垂直)
        x = joints[:, 0]  # 左右 -> x轴
        y = joints[:, 2]  # 深度 -> y轴  
        z = joints[:, 1]  # 高度 -> z轴（垂直显示）
        
        # 更新散点图
        coords = np.c_[x, y, z]
        scatter._offsets3d = (x, y, z)
        
        # 更新骨骼连接线
        for bone_idx, bone in enumerate(NTU_SKELETON_BONES):
            start_idx, end_idx = bone[0] - 1, bone[1] - 1
            if start_idx < len(joints) and end_idx < len(joints):
                lines[bone_idx].set_data([x[start_idx], x[end_idx]], 
                                         [y[start_idx], y[end_idx]])
                lines[bone_idx].set_3d_properties([z[start_idx], z[end_idx]])
        
        # 更新标题
        progress = (frame_idx + 1) / total_frames * 100
        title_text.set_text(f'Frame {frame_idx + 1}/{total_frames} ({progress:.1f}%)')
        
        return [scatter] + lines + [title_text]
    
    # 创建动画
    print(f"生成动画... (每{1/fps:.2f}秒一帧)")
    anim = FuncAnimation(fig, update, init_func=init, frames=total_frames, 
                        interval=1000/fps, blit=True, repeat=True)
    
    # 保存为GIF
    print(f"保存为GIF文件: {output_path}")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    
    print(f"✓ 动作视频已保存至: {output_path}")
    plt.close()


def visualize_random_video(dataset_path, num_samples=1, output_dir='skeleton_videos', fps=10):
    """随机生成视频"""
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
        output_path = os.path.join(output_dir, f'sample_{i+1}_{filename[:-9]}.gif')
        
        print(f"\n处理样本 {i+1}/{num_samples}: {filename}")
        try:
            create_skeleton_video(file_path, output_path, fps=fps, labels=labels)
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成NTU RGB+D骨架动作视频')
    parser.add_argument('--file', type=str, default=None, 
                        help='单个.skeleton文件路径')
    parser.add_argument('--dataset', type=str, 
                        default='./datasets/nturgbd_raw/',
                        help='数据集目录路径')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='随机生成的视频样本数量')
    parser.add_argument('--output', type=str, default='skeleton_video.gif',
                        help='输出视频路径')
    parser.add_argument('--fps', type=int, default=10,
                        help='视频帧率')
    
    args = parser.parse_args()
    
    if args.file:
        # 生成单个文件的视频
        labels = load_action_labels()
        create_skeleton_video(args.file, args.output, fps=args.fps, labels=labels)
    else:
        # 随机生成多个样本的视频
        visualize_random_video(args.dataset, args.num_samples, fps=args.fps)
