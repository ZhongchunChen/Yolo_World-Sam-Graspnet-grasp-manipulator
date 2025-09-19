import os
import sys
import numpy as np

# 修复 np.float 不存在的问题
if not hasattr(np, "float"):
    np.float = float
# 修复 np.maximum_sctype 不存在的问题
if not hasattr(np, "maximum_sctype"):
    np.maximum_sctype = lambda t: np.finfo(np.float64).max
    
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image
import spatialmath as sm

import cv2
import mujoco

from graspnetAPI import GraspGroup



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector


from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

from vision_processor import VisionProcessor
from trajectory_executor import TrajectoryExecutor
from grasp_predictor import GraspPredictor
from grasp_executor import GraspExecutor

# ================= 数据处理并生成输入 ====================
# def get_and_process_data(color_path, depth_path, mask_path):
#     """
#     根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
#     """
# #---------------------------------------
#     # 1. 加载 color（可能是路径，也可能是数组）
#     if isinstance(color_path, str):
#         color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
#     elif isinstance(color_path, np.ndarray):
#         color = color_path.astype(np.float32)
#         color /= 255.0
#     else:
#         raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

#     # 2. 加载 depth（可能是路径，也可能是数组）
#     if isinstance(depth_path, str):
#         depth_img = Image.open(depth_path)
#         depth = np.array(depth_img)
#     elif isinstance(depth_path, np.ndarray):
#         depth = depth_path
#     else:
#         raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

#     # 3. 加载 mask（可能是路径，也可能是数组）
#     if isinstance(mask_path, str):
#         workspace_mask = np.array(Image.open(mask_path))
#     elif isinstance(mask_path, np.ndarray):
#         workspace_mask = mask_path
#     else:
#         raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

#     # print("\n=== 尺寸验证 ===")
#     # print("深度图尺寸:", depth.shape)
#     # print("颜色图尺寸:", color.shape[:2])
#     # print("工作空间尺寸:", workspace_mask.shape)

#     # 构造相机内参矩阵
#     height = color.shape[0]
#     width = color.shape[1]
#     fovy = np.pi / 4 # 定义的仿真相机
#     focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（基于垂直视场角fovy和高度height）
#     c_x = width / 2.0   # 水平中心
#     c_y = height / 2.0  # 垂直中心
#     intrinsic = np.array([
#         [focal, 0.0, c_x],    
#         [0.0, focal, c_y],   
#         [0.0, 0.0, 1.0]
#     ])
#     factor_depth = 1.0  # 深度因子，根据实际数据调整

#     # 利用深度图生成点云 (H,W,3) 并保留组织结构
#     camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
#     cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

#     # mask = depth < 2.0
#     mask = (workspace_mask > 0) & (depth < 2.0)
#     cloud_masked = cloud[mask]
#     color_masked = color[mask]
#     # print(f"mask过滤后的点云数量 (color_masked): {len(color_masked)}") # 在采样前打印原始过滤后的点数

#     NUM_POINT = 3000 # 10000或5000
#     # 如果点数足够，随机采样NUM_POINT个点（不重复）
#     if len(cloud_masked) >= NUM_POINT:
#         idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
#     # 如果点数不足，先保留所有点，再随机重复补足NUM_POINT个点
#     else:
#         idxs1 = np.arange(len(cloud_masked))
#         idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
#         idxs = np.concatenate([idxs1, idxs2], axis=0)
    
#     cloud_sampled = cloud_masked[idxs]
#     color_sampled = color_masked[idxs] # 提取点云和颜色

#     cloud_o3d = o3d.geometry.PointCloud()
#     cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
#     cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
#     # end_points = {'point_clouds': cloud_sampled}

#     end_points = dict()
#     end_points['point_clouds'] = cloud_sampled
#     end_points['cloud_colors'] = color_sampled

#     return end_points, cloud_o3d

# =================== 获取抓取预测 ====================
# 已迁移到 grasp_predictor.py 中的 GraspPredictor 类
# def generate_grasps(end_points, cloud, visual=False):
#     """
#     主推理流程：
#     0. 数据处理并生成输入
#     1. 加载网络
#     2. 前向推理（进行抓取预测解码）
#     3. 碰撞检测
#     4. NMS 去重 + 按置信度/得分排序（降序）
#     5. 对抓取预测进行垂直角度筛选
#     """

#     # 1. 加载网络
#     net = GraspNet(input_feature_dim=0, 
#                    num_view=300, 
#                    num_angle=12, 
#                    num_depth=4,
#                    cylinder_radius=0.05, 
#                    hmin=-0.02, 
#                    hmax_list=[0.01, 0.02, 0.03, 0.04], 
#                    is_training=False)
#     net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
#     checkpoint = torch.load('./logs/log_rs/checkpoint-rs.tar') # checkpoint_path
#     net.load_state_dict(checkpoint['model_state_dict'])
#     net.eval()

#     # 2. 前向推理
#     with torch.no_grad():
#         end_points = net(end_points)
#         grasp_preds = pred_decode(end_points)
#     gg = GraspGroup(grasp_preds[0].detach().cpu().numpy()) 

#     # 3. 碰撞检测
#     COLLISION_THRESH = 0.01
#     if COLLISION_THRESH > 0:
#         voxel_size = 0.01
#         collision_thresh = 0.01
#         mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud.points), voxel_size=voxel_size)
#         collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
#         gg = gg[~collision_mask]

#     # 4. NMS 去重 + 按置信度/得分排序（降序）
#     gg.nms().sort_by_score()

#     # 5. 返回抓取得分最高的抓取（对抓取预测的接近方向进行垂直角度限制）
#     # 将 gg 转换为普通列表
#     all_grasps = list(gg)
#     vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面） np.array([0, 0, 1])
#     angle_threshold = np.deg2rad(30)  # 30度的弧度值 np.deg2rad(30)
#     filtered = []
#     for grasp in all_grasps:
#         # 抓取的接近方向取 grasp.rotation_matrix 的第三列[:, 0]
#         approach_dir = grasp.rotation_matrix[:, 0]
#         # 计算夹角：cos(angle)=dot(approach_dir, vertical)
#         cos_angle = np.dot(approach_dir, vertical)
#         cos_angle = np.clip(cos_angle, -1.0, 1.0)
#         angle = np.arccos(cos_angle)
#         if angle < angle_threshold:
#             filtered.append(grasp)
#     if len(filtered) == 0:
#         print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
#         filtered = all_grasps
#     # else:
#         print(f"\nFiltered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")

#     # 对过滤后的抓取根据 score 排序（降序）
#     filtered.sort(key=lambda g: g.score, reverse=True)

#     # 取前20个抓取（如果少于20个，则全部使用）
#     top_grasps = filtered[:20]
#     # top_grasps = filtered[:1]

#     # 可视化过滤后的抓取，手动转换为 Open3D 物体
#     grippers = [g.to_open3d_geometry() for g in top_grasps]
#     # print(f"\nVisualizing top {len(top_grasps)} grasps after vertical filtering...")
#     # o3d.visualization.draw_geometries([cloud, *grippers])
#     # for gripper in grippers:
#     #     o3d.visualization.draw_geometries([cloud, gripper])
    
#     # 选择得分最高的抓取（filtered 列表已按得分降序排序）
#     best_grasp = top_grasps[0]
#     best_translation = best_grasp.translation
#     best_rotation = best_grasp.rotation_matrix
#     best_width = best_grasp.width

#     # 创建一个新的 GraspGroup 并添加最佳抓取
#     new_gg = GraspGroup()            # 初始化空的 GraspGroup
#     new_gg.add(best_grasp)           # 添加最佳抓取
#     if visual:
#         grippers = new_gg.to_open3d_geometry_list()
#         o3d.visualization.draw_geometries([cloud, *grippers])
#     return new_gg
#     # return best_translation, best_rotation, best_width



# ================= 仿真执行抓取动作 ====================
# 已迁移到 grasp_executor.py 中的 GraspExecutor 类
# def execute_grasp(env, gg):
#     """
#     执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。

#     参数:
#     env (UR5GraspEnv): 机器人环境对象。
#     gg (GraspGroup): 抓取预测结果。
#     """
#     # 创建轨迹执行器
#     executor = TrajectoryExecutor(env)
#     robot = env.robot

#     # 0.初始准备阶段 - 计算抓取位姿 T_wo（物体相对于世界坐标系的位姿）
#     n_wc = np.array([0.0, -1.0, 0.0])  # 相机朝向
#     o_wc = np.array([-1.0, 0.0, -0.5]) # 相机朝向
#     t_wc = np.array([0.85, 0.8, 1.6])  # 相机位置

#     T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
#     T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))
#     T_wo = T_wc * T_co

#     # 1.机器人运动到预抓取位姿
#     q0 = robot.get_joint()
#     q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
#     planner1 = executor.create_joint_trajectory(q0, q1, duration=1.0)
#     executor.execute_single_trajectory(planner1, 1.0)

#     # 2.接近抓取位姿 - 从预抓取位姿直线移动到抓取点附近
#     robot.set_joint(q1)
#     T1 = robot.get_cartesian()
#     T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)  # 沿负x方向偏移0.1m确保安全接近
#     planner2 = executor.create_cartesian_trajectory(T1, T2, duration=1.0)
#     executor.execute_single_trajectory(planner2, 1.0)

#     # 3.执行抓取 - 移动到精确抓取位姿并闭合夹爪
#     T3 = T_wo
#     planner3 = executor.create_cartesian_trajectory(T2, T3, duration=1.0)
#     executor.execute_single_trajectory(planner3, 1.0)
#     # 闭合夹爪抓取物体
#     executor.control_gripper(steps=1000, delta=0.2, target_range=(0, 255))

#     # 4.提起物体 - 垂直提升避免碰撞桌面
#     T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3
#     planner4 = executor.create_cartesian_trajectory(T3, T4, duration=1.0)

#     # 5.水平移动物体 - 移动到目标放置位置
#     T5 = sm.SE3.Trans(1.4, 0.3, T4.t[2]) * sm.SE3(sm.SO3(T4.R))
#     planner5 = executor.create_cartesian_trajectory(T4, T5, duration=1.0)

#     # 6.放置物体 - 垂直下降到接触面
#     T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5
#     planner6 = executor.create_cartesian_trajectory(T5, T6, duration=1.0)

#     # 批量执行提起、移动、放置轨迹
#     executor.execute_trajectory([planner4, planner5, planner6], [1.0, 1.0, 1.0])
#     # 打开夹爪释放物体
#     executor.control_gripper(steps=1000, delta=-0.2, target_range=(0, 255))

#     # 7.抬起夹爪 - 避免碰撞物体
#     T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
#     planner7 = executor.create_cartesian_trajectory(T6, T7, duration=1.0)
#     executor.execute_single_trajectory(planner7, 1.0)

#     # 8.回到初始位置 - 完成整个任务
#     q8 = robot.get_joint()
#     planner8 = executor.create_joint_trajectory(q8, q0, duration=1.0)
#     executor.execute_single_trajectory(planner8, 1.0)


if __name__ == '__main__':
    
    env = UR5GraspEnv()
    env.reset()
    
    # 初始化视觉处理器
    vision_processor = VisionProcessor(
        yolo_model_path="yolov8s-world.pt",
        sam_model_path="sam_b.pt", 
        conf_threshold=0.25,
        fovy=np.pi/4,
        depth_factor=1.0
    )
    
    # 初始化抓取预测器
    grasp_predictor = GraspPredictor(
        checkpoint_path='./logs/log_rs/checkpoint-rs.tar',
        collision_thresh=0.01,
        angle_threshold_deg=30,
        top_k_grasps=20
    )
    
    # 初始化抓取执行器
    grasp_executor = GraspExecutor(
        env=env,
        camera_position=[0.85, 0.8, 1.6],
        approach_offset=0.1,
        lift_height=0.3,
        place_position=[1.4, 0.3]
    )

    n = 4 # 循环次数，连续抓取物体
    for iteration in range(n): 
        print(f"\n=== Starting iteration {iteration + 1}/{n} ===")
        
        # # 重置视觉处理器状态以避免设备冲突
        # if iteration > 0:
        #     print("Resetting vision processor to avoid device conflicts...")
        #     vision_processor.reset_models()

        for i in range(500): # 1000
            env.step()
        
        # 1. 获取图像和深度图
        imgs = env.render()
        color_img_path = imgs['img'] # MuJoCo 渲染的是 RGB
        depth_img_path = imgs['depth']

        # 将MuJoCo渲染的是RGB转化为OpenCV默认使用BGR颜色空间
        color_img_path = cv2.cvtColor(color_img_path, cv2.COLOR_RGB2BGR)
        # 保存/查看图片
        # cv2.imwrite('color_img_path.jpg', color_img_path)
        # cv2.imshow('color', color_img_path)
        # cv2.waitKey(0)
        
        # 2. 视觉分割图像
        mask_img_path = vision_processor.segment_image(color_img_path)

        # 3. 获取物体的点云数据
        end_points, cloud_o3d = vision_processor.get_and_process_data(color_img_path, depth_img_path, mask_img_path)

        # 4. 获取抓取点对应的夹爪姿态
        gg = grasp_predictor.predict_grasps(end_points, cloud_o3d, visual=True)

        # 5. 仿真执行抓取
        grasp_executor.execute_pick_and_place(gg, return_home=True)

    env.close()
