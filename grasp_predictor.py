import numpy as np
import torch
import open3d as o3d
from graspnetAPI import GraspGroup

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector


class GraspPredictor:
    """抓取预测器，封装GraspNet模型加载和抓取预测逻辑"""
    
    def __init__(self, 
                 checkpoint_path='./logs/log_rs/checkpoint-rs.tar',
                 input_feature_dim=0,
                 num_view=300,
                 num_angle=12,
                 num_depth=4,
                 cylinder_radius=0.05,
                 hmin=-0.02,
                 hmax_list=[0.01, 0.02, 0.03, 0.04],
                 collision_thresh=0.01,
                 voxel_size=0.01,
                 approach_dist=0.05,
                 angle_threshold_deg=30,
                 top_k_grasps=20):
        """
        初始化抓取预测器
        
        Args:
            checkpoint_path: 模型权重文件路径
            input_feature_dim: 输入特征维度
            num_view: 视角数量
            num_angle: 角度数量
            num_depth: 深度数量
            cylinder_radius: 圆柱体半径
            hmin: 最小高度
            hmax_list: 最大高度列表
            collision_thresh: 碰撞检测阈值
            voxel_size: 体素大小
            approach_dist: 接近距离
            angle_threshold_deg: 垂直角度阈值（度）
            top_k_grasps: 返回前k个抓取
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        self.approach_dist = approach_dist
        self.angle_threshold = np.deg2rad(angle_threshold_deg)
        self.top_k_grasps = top_k_grasps
        
        # 初始化并加载网络
        self.net = GraspNet(
            input_feature_dim=input_feature_dim,
            num_view=num_view,
            num_angle=num_angle,
            num_depth=num_depth,
            cylinder_radius=cylinder_radius,
            hmin=hmin,
            hmax_list=hmax_list,
            is_training=False
        )
        self.net.to(self.device)
        
        # 加载预训练权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        
        print(f"GraspPredictor initialized with device: {self.device}")
    
    def predict_grasps(self, end_points, cloud, visual=False):
        """
        预测抓取姿态
        
        Args:
            end_points: 点云数据字典
            cloud: Open3D点云对象
            visual: 是否可视化结果
            
        Returns:
            GraspGroup: 最佳抓取预测结果
        """
        # 1. 前向推理
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        
        # 2. 碰撞检测
        if self.collision_thresh > 0:
            gg = self._collision_detection(gg, cloud)
        
        # 3. NMS去重 + 按置信度排序
        gg.nms().sort_by_score()
        
        # 4. 垂直角度筛选
        filtered_grasps = self._filter_by_vertical_angle(gg)
        
        # 5. 选择最佳抓取
        best_grasp_group = self._select_best_grasp(filtered_grasps, cloud, visual)
        
        return best_grasp_group
    
    def _collision_detection(self, grasp_group, cloud):
        """执行碰撞检测"""
        mfcdetector = ModelFreeCollisionDetector(
            np.asarray(cloud.points), 
            voxel_size=self.voxel_size
        )
        collision_mask = mfcdetector.detect(
            grasp_group, 
            approach_dist=self.approach_dist, 
            collision_thresh=self.collision_thresh
        )
        return grasp_group[~collision_mask]
    
    def _filter_by_vertical_angle(self, grasp_group):
        """根据垂直角度筛选抓取"""
        all_grasps = list(grasp_group)
        vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面）
        filtered = []
        
        for grasp in all_grasps:
            # 抓取的接近方向
            approach_dir = grasp.rotation_matrix[:, 0]
            # 计算与垂直方向的夹角
            cos_angle = np.dot(approach_dir, vertical)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            if angle < self.angle_threshold:
                filtered.append(grasp)
        
        if len(filtered) == 0:
            print(f"\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
            filtered = all_grasps
        else:
            print(f"\nFiltered {len(filtered)} grasps within ±{np.rad2deg(self.angle_threshold):.0f}° of vertical out of {len(all_grasps)} total predictions.")
        
        return filtered
    
    def _select_best_grasp(self, filtered_grasps, cloud, visual=False):
        """选择最佳抓取并返回GraspGroup"""
        # 按得分排序
        filtered_grasps.sort(key=lambda g: g.score, reverse=True)
        
        # 取前k个抓取
        top_grasps = filtered_grasps[:min(len(filtered_grasps), self.top_k_grasps)]
        
        # 选择得分最高的抓取
        best_grasp = top_grasps[0]
        
        # 创建新的GraspGroup并添加最佳抓取
        result_gg = GraspGroup()
        result_gg.add(best_grasp)
        
        # 可视化
        if visual:
            grippers = result_gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])
        
        return result_gg
    
    def update_parameters(self, **kwargs):
        """更新预测参数"""
        for key, value in kwargs.items():
            if key == 'angle_threshold_deg':
                self.angle_threshold = np.deg2rad(value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter {key}")
