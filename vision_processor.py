import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
from PIL import Image
import logging
import open3d as o3d
from data_utils import CameraInfo, create_point_cloud_from_depth_image
# 禁用 Ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)


class VisionProcessor:
    """视觉处理器类，集成YOLO World检测和SAM分割功能"""
    
    def __init__(self, 
                 yolo_model_path="yolov8s-world.pt", 
                 sam_model_path="sam_b.pt",
                 conf_threshold=0.25,
                 fovy=np.pi/4,
                 depth_factor=1.0,
                 device="auto"):
        """
        初始化视觉处理器
        
        Args:
            yolo_model_path: YOLO World模型路径
            sam_model_path: SAM模型路径  
            conf_threshold: 置信度阈值
            fovy: 相机垂直视场角（弧度），默认45度
            depth_factor: 深度因子，默认1.0
            device: 计算设备，"auto"自动选择，"cpu"或"cuda:0"等
        """
        self.yolo_model_path = yolo_model_path
        self.sam_model_path = sam_model_path
        self.conf_threshold = conf_threshold
        self.fovy = fovy
        self.depth_factor = depth_factor
        self.device = self._resolve_device(device)
        
        # 延迟加载模型，避免初始化时的开销
        self._yolo_model = None
        self._sam_predictor = None
        
    def _resolve_device(self, device):
        """解析设备类型"""
        if device == "auto":
            return 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return device
        
    @property
    def yolo_model(self):
        """懒加载YOLO模型"""
        if self._yolo_model is None:
            self._yolo_model = YOLO(self.yolo_model_path)
            # 确保模型在指定的设备上
            self._yolo_model.to(self.device)
        return self._yolo_model
        
    @property 
    def sam_predictor(self):
        """懒加载SAM预测器"""
        if self._sam_predictor is None:
            self._sam_predictor = self._create_sam_predictor()
        return self._sam_predictor
        
    def _create_sam_predictor(self):
        """创建SAM预测器"""
        overrides = dict(
            task='segment',
            mode='predict',
            model=self.sam_model_path,
            conf=self.conf_threshold,
            save=False
        )
        return SAMPredictor(overrides=overrides)
        
    def _set_target_classes(self, target_classes):
        """设置YOLO World模型检测的目标类别"""
        if isinstance(target_classes, str):
            target_classes = [target_classes]
        
        # 确保模型在指定的设备上
        self.yolo_model.to(self.device)
        
        try:
            self.yolo_model.set_classes(target_classes)
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                # 设备不匹配时重新初始化模型
                print("Device mismatch detected, reinitializing YOLO model...")
                self._yolo_model = None  # 强制重新加载
                self.yolo_model.set_classes(target_classes)
            else:
                raise e

    def detect_objects(self, image_or_path, target_class=None):
        """
        使用YOLO World检测物体
        
        Args:
            image_or_path: 图像路径(str)或图像数组(np.ndarray)
            target_class: 目标类别名称
            
        Returns:
            tuple: (检测框列表, 可视化图像)
        """
        try:
            if target_class:
                self._set_target_classes(target_class)
                
            # YOLO推理
            results = self.yolo_model.predict(image_or_path)
            boxes = results[0].boxes
            vis_img = results[0].plot()
            
            # 提取有效检测结果
            valid_detections = []
            if boxes is not None:
                for box in boxes:
                    if box.conf.item() > self.conf_threshold:
                        valid_detections.append({
                            "xyxy": box.xyxy[0].tolist(),
                            "conf": box.conf.item(),
                            "cls": results[0].names[box.cls.item()]
                        })
                        
            return valid_detections, vis_img
            
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                print("Device conflict detected, resetting YOLO model...")
                self._yolo_model = None  # 强制重新加载
                return self.detect_objects(image_or_path, target_class)  # 递归重试
            else:
                raise e

    def _process_sam_results(self, sam_results):
        """
        处理SAM分割结果，提取掩码和中心点
        
        Args:
            sam_results: SAM模型的输出结果
            
        Returns:
            tuple: (中心点坐标, 二值掩码)
        """
        if not sam_results or not sam_results[0].masks:
            return None, None
            
        # 获取第一个掩码并二值化
        mask = sam_results[0].masks.data[0].cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255
        
        # 查找轮廓并计算中心点
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
            
        # 计算质心
        M = cv2.moments(contours[0])
        if M["m00"] == 0:
            return None, mask
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), mask
        
    def _get_user_click_point(self, vis_img):
        """获取用户点击的点坐标"""
        print("No detections - click on target object")
        cv2.imshow('Select Object', vis_img)
        
        point = []
        clicked = False
        
        def click_handler(event, x, y, flags, param):
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Clicked at ({x}, {y})")
                point.extend([x, y])
                clicked = True
                
        cv2.setMouseCallback('Select Object', click_handler)
        print("Waiting for user click...")
        
        # 等待点击或ESC退出
        while not clicked:
            key = cv2.waitKey(10)
            if key == 27:  # ESC键
                cv2.destroyAllWindows()
                raise ValueError("User cancelled selection")
                
        cv2.destroyAllWindows()
        return point if len(point) == 2 else None
        
    def segment_image(self, image_path, target_class=None, output_mask='mask1.png', auto_mode=True):
        """
        完整的图像分割流程
        
        Args:
            image_path: 图像路径(str)或图像数组(BGR格式)
            target_class: 目标类别，如果为None则提示用户输入
            output_mask: 输出掩码文件名
            auto_mode: 是否自动选择最高置信度检测结果
            
        Returns:
            np.ndarray: 分割掩码 或 None
        """
        # 1) 获取目标类别
        if target_class is None and auto_mode:
            target_class = input("\n===============\nEnter class name: ").strip()
            
        # 2) YOLO检测
        detections, vis_img = self.detect_objects(image_path, target_class)
        
        # 保存检测可视化结果
        cv2.imwrite('detection_visualization.jpg', vis_img)
        
        # 3) 准备SAM输入图像(RGB格式)
        if isinstance(image_path, str):
            bgr_img = cv2.imread(image_path)
            if bgr_img is None:
                raise ValueError(f"Failed to read image from path: {image_path}")
            image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        else:
            # 假设是BGR数组
            image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
            
        # 4) 设置SAM图像
        self.sam_predictor.set_image(image_rgb)
        
        # 5) 执行分割
        center, mask = None, None
        
        if detections and auto_mode:
            # 自动模式：选择最高置信度
            best_det = max(detections, key=lambda x: x["conf"])
            sam_results = self.sam_predictor(bboxes=[best_det["xyxy"]])
            center, mask = self._process_sam_results(sam_results)
            print(f"Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
        else:
            # 手动模式：用户点击
            point = self._get_user_click_point(vis_img)
            if point:
                sam_results = self.sam_predictor(points=[point], labels=[1])
                center, mask = self._process_sam_results(sam_results)
            else:
                raise ValueError("No selection made")
                
        # 6) 保存掩码
        if mask is not None:
            cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        else:
            print("[WARNING] Could not generate mask")
            
        return mask
    def _path_to_array(self, path_or_array):
        """
        将图像路径转换为NumPy数组
        
        Args:
            path_or_array: 图像路径(str)或图像数组(np.ndarray)
        Returns:
            np.ndarray: 图像数组
        """
        if isinstance(path_or_array, str):
            img = np.array(Image.open(path_or_array))
        elif isinstance(path_or_array, np.ndarray):
            img = path_or_array
        else:
            raise TypeError("输入既不是字符串路径也不是 NumPy 数组！")
        return img

    def _create_camera(self, width, height, fovy, factor_depth):
        """
        创建相机内参矩阵
        
        Args:
            width: 图像宽度
            height: 图像高度
            fovy: 垂直视场角（弧度）
            factor_depth: 深度因子

        Returns:
            CameraInfo: 相机信息对象
        """
        focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算
        c_x = width / 2.0   # 水平中心
        c_y = height / 2.0  # 垂直中心
        intrinsic = np.array([
            [focal, 0.0, c_x],    
            [0.0, focal, c_y],   
            [0.0, 0.0, 1.0]
        ])
        camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        return camera

    def _create_point_clouds_from_depth(self, depth, camera):
        """
        从深度图生成点云
        
        Args:
            depth: 深度图 (H, W)
            camera: CameraInfo 对象
            
        Returns:
            np.ndarray: 点云 (H, W, 3)
        """
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        return cloud

    def _filter_by_mask(self, cloud, color, workspace_mask, depth_image, depth_threshold=2.0):
        """
        使用掩码和深度阈值过滤点云和颜色

        Args:
            cloud: 点云 (H, W, 3)
            color: 颜色 (H, W, 3)
            workspace_mask: 二值掩码 (H, W)
            depth_image: 深度图像 (H, W)
            depth_threshold: 深度阈值，过滤掉大于此值的点

        Returns:
            tuple: (过滤后的点云, 过滤后的颜色)
        """
        # 创建综合掩码：工作空间内且深度小于阈值的点
        valid_mask = (workspace_mask > 0) & (depth_image < depth_threshold)
        
        # 应用掩码过滤点云和颜色
        cloud_masked = cloud[valid_mask]
        color_masked = color[valid_mask]
        
        print(f"Filtered points: {len(cloud_masked)} out of {cloud.size // 3} total points")
        return cloud_masked, color_masked

    def _random_sample_points(self, cloud_masked, color_masked, num_points=3000):
        """
        随机采样指定数量的点云和颜色
        
        Args:
            cloud_masked: 过滤后的点云 (N, 3)
            color_masked: 过滤后的颜色 (N, 3)
            num_points: 采样点数
            
        Returns:
            tuple: (采样后的点云, 采样后的颜色)
        """
        # 检查是否有有效点云数据
        if len(cloud_masked) == 0:
            raise ValueError("No valid points found after masking. Check your depth image and workspace mask.")
        
        if len(cloud_masked) >= num_points:
            # 点数足够，随机采样不重复
            idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
        else:
            # 点数不足，先保留所有点，再随机重复补足
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        return cloud_sampled, color_sampled

    def _create_open3d_pointcloud(self, cloud_points, cloud_colors):
        """
        创建Open3D点云对象
        
        Args:
            cloud_points: 点云坐标 (N, 3)
            cloud_colors: 点云颜色 (N, 3)
            
        Returns:
            o3d.geometry.PointCloud: Open3D点云对象
        """
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_points.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(cloud_colors.astype(np.float32))
        return cloud_o3d

    def _create_pytorch_endpoints(self, cloud_sampled, color_sampled):
        """
        将采样后的点云转换为PyTorch张量并创建end_points字典
        
        Args:
            cloud_sampled: 采样后的点云 (N, 3)
            color_sampled: 采样后的颜色 (N, 3)  
            
        Returns:
            dict: end_points字典，包含PyTorch张量格式的点云数据
        """
        # 使用实例的设备设置
        device = torch.device(self.device)
        
        # 转换为PyTorch张量：添加batch维度 (1, N, 3)
        cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
        
        # 创建end_points字典
        end_points = {
            'point_clouds': cloud_tensor,
            'cloud_colors': color_sampled
        }
        
        return end_points

    def get_and_process_data(self, color_path, depth_path, mask_path, NUM_POINT = 3000):
        """
        根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据

        Args:
            color_path: RGB图像路径或数组 (H, W, 3)
            depth_path: 深度图像路径或数组 (H, W)
            mask_path: 掩码图像路径或数组 (H, W)
            NUM_POINT: 采样点数，默认3000(可选)

        Returns:
            tuple: (end_points字典, Open3D点云对象)
        """
        # 1. 加载 color（可能是路径，也可能是数组）
        color = self._path_to_array(color_path)
        # 确保颜色数据在[0,1]范围内
        if color.dtype == np.uint8:
            color = color.astype(np.float32) / 255.0

        # 2. 加载 depth（可能是路径，也可能是数组）
        depth = self._path_to_array(depth_path)

        # 3. 加载 mask（可能是路径，也可能是数组）
        workspace_mask = self._path_to_array(mask_path)

        # print("\n=== 数据验证 ===")
        # print(f"颜色图尺寸: {color.shape}, 数据类型: {color.dtype}, 范围: [{color.min():.3f}, {color.max():.3f}]")
        # print(f"深度图尺寸: {depth.shape}, 数据类型: {depth.dtype}, 范围: [{depth.min():.3f}, {depth.max():.3f}]")
        # print(f"掩码图尺寸: {workspace_mask.shape}, 数据类型: {workspace_mask.dtype}, 唯一值: {np.unique(workspace_mask)}")

        # 构造相机
        height = color.shape[0]
        width = color.shape[1]
        camera = self._create_camera(width, height, self.fovy, self.depth_factor)
        # 利用深度图生成点云 (H,W,3) 并保留组织结构
        cloud = self._create_point_clouds_from_depth(depth, camera)

        # 使用掩码和深度阈值过滤点云
        cloud_masked, color_masked = self._filter_by_mask(cloud, color, workspace_mask, depth, depth_threshold=2.0)
        # print(f"mask过滤后的点云数量 (color_masked): {len(color_masked)}") # 在采样前打印原始过滤后的点数

        cloud_sampled, color_sampled = self._random_sample_points(cloud_masked, color_masked, num_points=NUM_POINT)

        # 创建Open3D点云对象
        cloud_o3d = self._create_open3d_pointcloud(cloud_masked, color_masked)

        # 转换为PyTorch张量并创建end_points字典(包含采样后的点云)
        end_points = self._create_pytorch_endpoints(cloud_sampled, color_sampled)

        return end_points, cloud_o3d

    def reset_models(self):
        """重置模型状态，解决设备不匹配问题"""
        self._yolo_model = None
        self._sam_predictor = None

if __name__ == '__main__':
    # 示例：指定使用CPU
    # processor = VisionProcessor()
    # 或者指定使用GPU
    # processor = VisionProcessor(device="cuda:0")
    # 或者自动选择设备（默认）
    processor = VisionProcessor(device="auto")
    
    print(f"Using device: {processor.device}")
    seg_mask = processor.segment_image('color_img_path.jpg', target_class="apple")
    print("Class-based result:", seg_mask.shape if seg_mask is not None else None)

