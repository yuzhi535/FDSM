import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv

# NTU RGB+D 骨架连接关系（25个关节点）
NTU_SKELETON_BONES = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
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
    except:
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
                    x, y, z = float(joint_info[0]), float(joint_info[1]), float(joint_info[2])
                    joints.append([x, y, z])
                
                frame_bodies.append(np.array(joints))
            
            frames_data.append(frame_bodies)
    
    return frames_data


def save_single_frame(skeleton, output_path, coord_range, title="", action_label=""):
    """保存单帧骨架为独立图片"""
    if len(skeleton) == 0:
        return
    
    joints = skeleton[0]
    
    # 坐标映射
    x = joints[:, 0]
    y = joints[:, 2]
    z = joints[:, 1]
    
    # 创建图表
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_alpha(0.0)  # 设置图形背景透明
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_alpha(0.0)  # 设置坐标系背景透明
    
    # 绘制关节点 - 增大尺寸
    ax.scatter(x, y, z, c='red', marker='o', s=150, alpha=0.8)
    
    # 绘制骨骼连接 - 增粗线条
    for bone in NTU_SKELETON_BONES:
        start_idx, end_idx = bone[0] - 1, bone[1] - 1
        if start_idx < len(joints) and end_idx < len(joints):
            ax.plot([x[start_idx], x[end_idx]],
                   [y[start_idx], y[end_idx]],
                   [z[start_idx], z[end_idx]],
                   'b-', linewidth=5, alpha=0.6)
    
    # 隐藏坐标轴
    ax.set_axis_off()
    
    # 设置坐标范围
    if coord_range:
        x_range, y_range, z_range = coord_range
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
    
    # 设置视角
    ax.view_init(elev=15, azim=-75)
    
    # # 设置标题
    # if action_label:
    #     fig.suptitle(f'{action_label}\n{title}', fontsize=11, fontweight='bold')
    # else:
    #     fig.suptitle(title, fontsize=11, fontweight='bold')
    
    # 最小化所有边距，让骨架充满图片
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


def visualize_skeleton_frames(skeleton_file, output_dir, num_frames=6, labels=None):
    """将骨架序列保存为多张独立图片"""
    print(f"读取骨架文件: {skeleton_file}")
    frames_data = read_skeleton_file(skeleton_file)
    
    # 提取动作标签
    filename = os.path.basename(skeleton_file)
    base_name = filename.replace('.skeleton', '')
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
    
    # 计算整个序列的坐标范围
    all_joints = []
    for frame_idx in frame_indices:
        if len(frames_data[frame_idx]) > 0:
            all_joints.append(frames_data[frame_idx][0])
    
    if len(all_joints) > 0:
        all_joints = np.vstack(all_joints)
        # 极小的padding，最大化骨架显示
        padding = 0.02
        x_min, x_max = all_joints[:, 0].min() - padding, all_joints[:, 0].max() + padding
        y_min, y_max = all_joints[:, 2].min() - padding, all_joints[:, 2].max() + padding
        z_min, z_max = all_joints[:, 1].min() - padding, all_joints[:, 1].max() + padding
        coord_range = ([x_min, x_max], [y_min, y_max], [z_min, z_max])
    else:
        coord_range = None
    
    # 创建输出目录
    frame_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(frame_output_dir, exist_ok=True)
    
    # 保存每一帧
    for idx, frame_idx in enumerate(frame_indices):
        output_path = os.path.join(frame_output_dir, f'frame_{frame_idx+1:04d}.png')
        title = f'Frame {frame_idx+1}/{total_frames}'
        save_single_frame(frames_data[frame_idx], output_path, coord_range, title, action_label)
        print(f"  保存帧 {idx+1}/{len(frame_indices)}: {output_path}")
    
    print(f"✓ 所有帧已保存至: {frame_output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='将NTU RGB+D骨架序列保存为独立的帧图片')
    parser.add_argument('--file', type=str, required=True,
                        help='skeleton文件路径')
    parser.add_argument('--output-dir', type=str, default='./skeleton_frames',
                        help='输出目录')
    parser.add_argument('--num-frames', type=int, default=6,
                        help='要保存的帧数')
    
    args = parser.parse_args()
    
    # 加载标签
    labels = load_action_labels()
    
    # 生成独立帧图片
    visualize_skeleton_frames(args.file, args.output_dir, args.num_frames, labels)
