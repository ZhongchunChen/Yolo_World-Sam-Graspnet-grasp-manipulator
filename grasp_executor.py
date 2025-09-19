import numpy as np
import spatialmath as sm
from trajectory_executor import TrajectoryExecutor


class GraspExecutor:
    """抓取执行器，封装完整的抓取动作执行逻辑"""
    
    def __init__(self, env, 
                 # 相机参数
                 camera_position=[0.85, 0.8, 1.6],
                 camera_orientation_x=[0.0, -1.0, 0.0], 
                 camera_orientation_y=[-1.0, 0.0, -0.5],
                 # 机器人位置参数
                 pre_grasp_joint=[0.0, 0.0, np.pi/2, 0.0, -np.pi/2, 0.0],
                 # 运动参数
                 approach_offset=0.1,        # 接近偏移距离(m)
                 lift_height=0.3,           # 提升高度(m) 
                 place_position=[1.4, 0.3], # 放置位置[x, y]
                 drop_height=0.1,           # 下降高度(m)
                 retreat_height=0.1,        # 撤退高度(m)
                 # 时间参数
                 default_duration=1.0,      # 默认轨迹时间(s)
                 # 夹爪参数
                 gripper_close_steps=1000,
                 gripper_open_steps=1000,
                 gripper_delta=0.2,
                 gripper_range=(0, 255)):
        """
        初始化抓取执行器
        
        Args:
            env: 机器人环境对象
            camera_position: 相机世界坐标位置 [x, y, z]
            camera_orientation_x: 相机x轴方向向量
            camera_orientation_y: 相机y轴方向向量  
            pre_grasp_joint: 预抓取关节角度
            approach_offset: 接近目标时的安全偏移距离
            lift_height: 抓取后垂直提升高度
            place_position: 放置目标位置 [x, y]
            drop_height: 放置时下降高度
            retreat_height: 放置后撤退高度
            default_duration: 默认轨迹执行时间
            gripper_close_steps: 夹爪闭合步数
            gripper_open_steps: 夹爪打开步数
            gripper_delta: 夹爪每步变化量
            gripper_range: 夹爪控制范围
        """
        self.env = env
        self.robot = env.robot
        self.executor = TrajectoryExecutor(env)
        
        # 相机参数
        self.camera_position = np.array(camera_position)
        self.camera_orientation_x = np.array(camera_orientation_x) 
        self.camera_orientation_y = np.array(camera_orientation_y)
        
        # 机器人参数
        self.pre_grasp_joint = np.array(pre_grasp_joint)
        
        # 运动参数
        self.approach_offset = approach_offset
        self.lift_height = lift_height
        self.place_position = place_position
        self.drop_height = drop_height
        self.retreat_height = retreat_height
        self.default_duration = default_duration
        
        # 夹爪参数
        self.gripper_close_steps = gripper_close_steps
        self.gripper_open_steps = gripper_open_steps
        self.gripper_delta = gripper_delta
        self.gripper_range = gripper_range
        
    def _compute_world_grasp_pose(self, grasp_group):
        """计算世界坐标系下的抓取位姿"""
        # 相机到世界的变换矩阵
        T_wc = sm.SE3.Trans(self.camera_position) * sm.SE3(
            sm.SO3.TwoVectors(x=self.camera_orientation_x, y=self.camera_orientation_y)
        )
        
        # 抓取相对于相机的变换矩阵
        T_co = sm.SE3.Trans(grasp_group.translations[0]) * sm.SE3(
            sm.SO3.TwoVectors(
                x=grasp_group.rotation_matrices[0][:, 0], 
                y=grasp_group.rotation_matrices[0][:, 1]
            )
        )
        
        # 世界坐标系下的抓取位姿
        T_wo = T_wc * T_co
        return T_wo
        
    def _move_to_pre_grasp(self):
        """移动到预抓取位姿"""
        q0 = self.robot.get_joint()
        planner = self.executor.create_joint_trajectory(q0, self.pre_grasp_joint, self.default_duration)
        self.executor.execute_single_trajectory(planner, self.default_duration)
        
    def _approach_target(self, target_pose):
        """接近目标位置"""
        self.robot.set_joint(self.pre_grasp_joint)
        current_pose = self.robot.get_cartesian()
        
        # 沿负x方向偏移，确保安全接近
        approach_pose = target_pose * sm.SE3(-self.approach_offset, 0.0, 0.0)
        
        planner = self.executor.create_cartesian_trajectory(current_pose, approach_pose, self.default_duration)
        self.executor.execute_single_trajectory(planner, self.default_duration)
        return approach_pose
        
    def _execute_grasp(self, approach_pose, target_pose):
        """执行抓取动作"""
        # 移动到精确抓取位姿
        planner = self.executor.create_cartesian_trajectory(approach_pose, target_pose, self.default_duration)
        self.executor.execute_single_trajectory(planner, self.default_duration)
        
        # 闭合夹爪
        self.executor.control_gripper(
            steps=self.gripper_close_steps, 
            delta=self.gripper_delta, 
            target_range=self.gripper_range
        )
        
    def _lift_object(self, grasp_pose):
        """提起物体"""
        lift_pose = sm.SE3.Trans(0.0, 0.0, self.lift_height) * grasp_pose
        planner = self.executor.create_cartesian_trajectory(grasp_pose, lift_pose, self.default_duration)
        self.executor.execute_single_trajectory(planner, self.default_duration)
        return lift_pose
        
    def _move_to_place_position(self, current_pose):
        """移动到放置位置"""
        place_pose = sm.SE3.Trans(
            self.place_position[0], 
            self.place_position[1], 
            current_pose.t[2]
        ) * sm.SE3(sm.SO3(current_pose.R))
        
        planner = self.executor.create_cartesian_trajectory(current_pose, place_pose, self.default_duration)
        self.executor.execute_single_trajectory(planner, self.default_duration)
        return place_pose
        
    def _place_object(self, place_pose):
        """放置物体"""
        # 下降到接触面
        drop_pose = sm.SE3.Trans(0.0, 0.0, -self.drop_height) * place_pose
        planner = self.executor.create_cartesian_trajectory(place_pose, drop_pose, self.default_duration)
        self.executor.execute_single_trajectory(planner, self.default_duration)
        
        # 打开夹爪
        self.executor.control_gripper(
            steps=self.gripper_open_steps, 
            delta=-self.gripper_delta, 
            target_range=self.gripper_range
        )
        return drop_pose
        
    def _retreat(self, drop_pose):
        """撤退到安全位置"""
        retreat_pose = sm.SE3.Trans(0.0, 0.0, self.retreat_height) * drop_pose
        planner = self.executor.create_cartesian_trajectory(drop_pose, retreat_pose, self.default_duration)
        self.executor.execute_single_trajectory(planner, self.default_duration)
        
    def _return_to_home(self):
        """返回初始位置"""
        current_joint = self.robot.get_joint()
        # 定义一个默认的home位置，或者使用初始关节角度
        home_joint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 可以根据需要调整
        
        # 检查是否已经在home位置
        if np.allclose(current_joint, home_joint, atol=0.01):
            print("Already at home position")
            return
            
        planner = self.executor.create_joint_trajectory(current_joint, home_joint, self.default_duration)
        self.executor.execute_single_trajectory(planner, self.default_duration)
        
    def execute_pick_and_place(self, grasp_group, return_home=True):
        """
        执行完整的抓取和放置动作
        
        Args:
            grasp_group: GraspNet预测的抓取组
            return_home: 是否返回初始位置
            
        Returns:
            bool: 执行是否成功
        """
        try:
            print("Starting pick and place execution...")
            
            # 1. 计算世界坐标系下的抓取位姿
            target_pose = self._compute_world_grasp_pose(grasp_group)
            
            # 2. 移动到预抓取位姿
            print("Step 1: Moving to pre-grasp position...")
            self._move_to_pre_grasp()
            
            # 3. 接近目标
            print("Step 2: Approaching target...")
            approach_pose = self._approach_target(target_pose)
            
            # 4. 执行抓取
            print("Step 3: Executing grasp...")
            self._execute_grasp(approach_pose, target_pose)
            
            # 5. 提起物体
            print("Step 4: Lifting object...")
            lift_pose = self._lift_object(target_pose)
            
            # 6. 移动到放置位置
            print("Step 5: Moving to place position...")
            place_pose = self._move_to_place_position(lift_pose)
            
            # 7. 放置物体
            print("Step 6: Placing object...")
            drop_pose = self._place_object(place_pose)
            
            # 8. 撤退
            print("Step 7: Retreating...")
            self._retreat(drop_pose)
            
            # 9. 返回初始位置(可选)
            if return_home:
                print("Step 8: Returning home...")
                self._return_to_home()
            
            print("Pick and place execution completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during pick and place execution: {e}")
            return False
            
    def update_parameters(self, **kwargs):
        """更新执行参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Updated {key} to {value}")
            else:
                print(f"Warning: Unknown parameter {key}")
                
    def get_current_config(self):
        """获取当前配置"""
        return {
            'camera_position': self.camera_position.tolist(),
            'pre_grasp_joint': self.pre_grasp_joint.tolist(),
            'approach_offset': self.approach_offset,
            'lift_height': self.lift_height,
            'place_position': self.place_position,
            'default_duration': self.default_duration
        }
