import torch
import os
import cv2
from PIL import Image, ImageDraw
import numpy as np
from tqdm.auto import tqdm


from modules.mesh_fusion.render import (
    features_to_world_space_mesh,
    render_mesh,
    edge_threshold_filter,
    unproject_points,
)
from utils.common_utils import (
    visualize_depth_numpy,
    save_rgbd,
)

import time
from utils.camera_utils import *

import utils.functions as functions
from utils.functions import rot_x_world_to_cam, rot_y_world_to_cam, rot_z_world_to_cam, colorize_single_channel_image, write_video
from modules.equilib import equi2pers, cube2equi, equi2cube

from modules.geo_predictors.PanoFusionDistancePredictor import PanoFusionDistancePredictor
from modules.inpainters import PanoPersFusionInpainter
from modules.geo_predictors import PanoJointPredictor
from modules.mesh_fusion.sup_info import SupInfoPool
from kornia.morphology import erosion, dilation
from scene.arguments import GSParams, CameraParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.graphics import focal2fov
from utils.loss import l1_loss, ssim
from random import randint

@torch.no_grad()
class Pano2RoomPipeline(torch.nn.Module):
    def __init__(self, attempt_idx=""):
        super().__init__()

        # 渲染器设置
        self.blur_radius = 0  # 模糊半径，用于渲染时的平滑处理
        self.faces_per_pixel = 8  # 每个像素的面数，影响渲染质量
        self.fov = 90  # 视场角度
        self.R, self.T = torch.Tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]), torch.Tensor([[0., 0., 0.]])  # 旋转和平移矩阵
        self.pano_width, self.pano_height = 1024 * 2, 512 * 2  # 全景图尺寸
        self.H, self.W = 512, 512  # 输出图像尺寸
        self.device = "cuda:0"  # 使用的设备

        # 初始化渲染相关的张量
        self.rendered_depth = torch.zeros((self.H, self.W), device=self.device)  # 渲染的深度图
        self.inpaint_mask = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)  # 修复遮罩
        self.vertices = torch.empty((3, 0), device=self.device, requires_grad=False)  # 顶点坐标
        self.colors = torch.empty((3, 0), device=self.device, requires_grad=False)  # 顶点颜色
        self.faces = torch.empty((3, 0), device=self.device, dtype=torch.long, requires_grad=False)  # 面片索引
        self.pix_to_face = None  # 像素到面片的映射

        # 场景参数设置
        self.pose_scale = 0.6  # 姿态缩放因子
        self.pano_center_offset = (-0.2,0.3)  # 全景图中心偏移
        self.inpaint_frame_stride = 20  # 修复帧的步长

        # 创建输出目录
        self.setting = f""
        apply_timestamp = True
        if apply_timestamp:
            timestamp = str(int(time.time()))[-8:]
            self.setting += f"-{timestamp}"
        self.save_path = f'output/Pano2Room-results'  # 结果保存路径
        self.save_details = False  # 是否保存详细信息

        # 创建输出目录（如果不存在）
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print("makedir:", self.save_path)

        # 初始化相机变换矩阵
        self.world_to_cam = torch.eye(4, dtype=torch.float32, device=self.device)  # 世界坐标到相机坐标的变换矩阵
        self.cubemap_w2c_list = functions.get_cubemap_views_world_to_cam()  # 获取立方体贴图的视图变换矩阵列表

        # 加载必要的模块
        self.load_modules()

    def load_modules(self):
        """加载必要的模块
        - inpainter: 用于全景图修复的模块
        - geo_predictor: 用于几何预测的模块
        """
        self.inpainter = PanoPersFusionInpainter(save_path=self.save_path)
        self.geo_predictor = PanoJointPredictor(save_path=self.save_path)

    def project(self, world_to_cam):
        """将3D网格投影到2D视图并进行渲染
        Args:
            world_to_cam: 世界坐标到相机坐标的变换矩阵
        Returns:
            rendered_image_tensor: 渲染的图像张量
            rendered_image_pil: 渲染的PIL图像
        """
        # 投影网格并渲染（RGB、深度、遮罩）
        rendered_image_tensor, self.rendered_depth, self.inpaint_mask, self.pix_to_face, self.z_buf, self.mesh = render_mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_features=self.colors,
            H=self.H,
            W=self.W,
            fov_in_degrees=self.fov,
            RT=world_to_cam,
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel
        )
        # 使用遮罩处理渲染图像
        rendered_image_tensor = rendered_image_tensor * ~self.inpaint_mask

        # 将渲染结果转换为PIL图像格式
        rendered_image_pil = Image.fromarray((rendered_image_tensor.permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
        self.inpaint_mask_pil = Image.fromarray(self.inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")

        # 保存当前状态以便恢复
        self.inpaint_mask_restore = self.inpaint_mask
        self.inpaint_mask_pil_restore = self.inpaint_mask_pil

        return rendered_image_tensor[:3, ...], rendered_image_pil

    def render_pano(self, pose):
        """渲染全景图
        Args:
            pose: 相机姿态
        Returns:
            pano_rgb: 渲染的RGB全景图
            pano_depth: 渲染的深度全景图
            pano_mask: 渲染的遮罩全景图
        """
        cubemap_list = [] 
        # 遍历立方体贴图的六个面
        for cubemap_pose in self.cubemap_w2c_list:
            pose_tmp = pose.clone()
            pose_tmp = cubemap_pose.cuda() @ pose_tmp
            rendered_image_tensor, rendered_image_pil = self.project(pose_tmp.cuda())

            # 准备立方体贴图的各个通道
            rgb_CHW = rendered_image_tensor.squeeze(0).cuda()
            depth_CHW = self.rendered_depth.unsqueeze(0).cuda()
            distance_CHW = functions.depth_to_distance(depth_CHW)
            mask_CHW = self.inpaint_mask.unsqueeze(0).cuda()
            cubemap_list += [torch.cat([rgb_CHW, distance_CHW, mask_CHW], axis=0)]

        # 将立方体贴图转换为全景图
        torch.set_default_tensor_type('torch.FloatTensor')
        pano_rgbd = cube2equi(cubemap_list,
                                "list",
                                1024,2048) #CHW
        pano_rgb = pano_rgbd[:3,:,:]
        pano_depth =  pano_rgbd[3:4,:,:].squeeze(0)
        pano_mask =  pano_rgbd[4:,:,:].squeeze(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return pano_rgb, pano_depth, pano_mask  # CHW, HW, HW

    def rgbd_to_mesh(self, rgb, depth, world_to_cam=None, mask=None, pix_to_face=None, using_distance_map=False):
        """将RGBD图像转换为3D网格
        Args:
            rgb: RGB图像数据
            depth: 深度图数据
            world_to_cam: 世界坐标到相机坐标的变换矩阵（可选）
            mask: 遮罩图像（可选）
            pix_to_face: 像素到面片的映射（可选）
            using_distance_map: 是否使用距离图（默认False）
        """
        predicted_depth = depth.cuda()
        rgb = rgb.squeeze(0).cuda()
        if world_to_cam is None:
            world_to_cam = torch.eye(4, dtype=torch.float32)
        world_to_cam = world_to_cam.cuda()
        if pix_to_face is not None:
            self.pix_to_face = pix_to_face
        if mask is None:
            self.inpaint_mask = torch.ones_like(predicted_depth)
        else:
            self.inpaint_mask = mask

        if self.inpaint_mask.sum() == 0:
            return
        
        # 将���征转换为世界空间中的网格
        vertices, faces, colors = features_to_world_space_mesh(
            colors=rgb,
            depth=predicted_depth,
            fov_in_degrees=self.fov,
            world_to_cam=world_to_cam,
            mask=self.inpaint_mask,
            pix_to_face=self.pix_to_face,
            faces=self.faces,
            vertices=self.vertices,
            using_distance_map=using_distance_map,
            edge_threshold=0.05
        )

        # 更新面片索引
        faces += self.vertices.shape[1] 

        # 保存当前状态以便恢复
        self.vertices_restore = self.vertices.clone()
        self.colors_restore = self.colors.clone()
        self.faces_restore = self.faces.clone()

        # 更新网格数据
        self.vertices = torch.cat([self.vertices, vertices], dim=1)
        self.colors = torch.cat([self.colors, colors], dim=1)
        self.faces = torch.cat([self.faces, faces], dim=1)

    def find_depth_edge(self, depth, dilate_iter=0):
        """查找深度图的边缘
        Args:
            depth: 深度图数据
            dilate_iter: 膨胀迭代次数（默认0）
        Returns:
            edges: 边缘图像
        """
        # 将深度图归一化到0-255范围
        gray = (depth/depth.max() * 255).astype(np.uint8)
        # 使Canny算子检测边缘
        edges = cv2.Canny(gray, 60, 150)
        # 如果需要，对边缘进行膨胀处理
        if dilate_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
        return edges

    def pano_distance_to_mesh(self, pano_rgb, pano_distance, depth_edge_inpaint_mask, pose=None):
        """将全景距离图转换为3D网格
        Args:
            pano_rgb: 全景RGB图像
            pano_distance: 全景距离图
            depth_edge_inpaint_mask: 深度边缘修复遮罩
            pose: 相机姿态（可选）
        """
        self.rgbd_to_mesh(pano_rgb, pano_distance, mask=depth_edge_inpaint_mask, using_distance_map=True, world_to_cam=pose)
 
    def load_inpaint_poses(self):
        """加载用于修复的相机姿态
        Returns:
            pose_dict: 包含相机姿态的字典，格式为 {idx: pose}
        """
        # 渲染当前视角的全景图
        pano_rgb, pano_distance, pano_mask = self.render_pano(self.world_to_cam)

        pose_dict = {} # {idx:pose, ...} # pose are c2w
        key = 0

        # 按照指定步长采样相机姿态
        sampled_inpaint_poses = self.poses[::self.inpaint_frame_stride]
        for anchor_idx in range(len(sampled_inpaint_poses)):
            pose = torch.eye(4).float() # pano pose dosen't support rotations

            # 转换相机姿态
            pose_44 = sampled_inpaint_poses[anchor_idx].clone()
            pose_44 = pose_44.float()
            Rw2c = pose_44[:3,:3].cpu().numpy()  # 世界到相机的旋转矩阵
            Tw2c = pose_44[:3,3:].cpu().numpy()  # 世界到相机的平移向量
            yz_reverse = np.array([[1,0,0], [0,1,0], [0,0,1]])  # YZ轴反转矩阵
            
            # 计算相机到世界的变换
            Rc2w = np.matmul(yz_reverse, Rw2c).T  # 相机到世界的旋转矩阵
            Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))  # 相机到世界的平移向量
            Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)  # 组合旋转和平移
            Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)  # 添加齐次坐标
            
            # 更新姿态信息
            pose[:3, 3] = torch.tensor(Pc2w[:3, 3]).cuda().float()
            pose[:3, 3] *= -1
            pose_dict[key] = pose.clone()

            key += 1
        return pose_dict

    def stage_inpaint_pano_greedy_search(self, pose_dict): 
        """使用贪婪搜索策略进行全景图修复
        Args:
            pose_dict: 包含相机姿态的字典
        Returns:
            inpainted_panos_and_poses: 修复后的全景图和对应姿态的列表
        """
        print("stage_inpaint_pano_greedy_search")
        # 渲染初始全景图
        pano_rgb, pano_distance, pano_mask = self.render_pano(self.world_to_cam)

        inpainted_panos_and_poses = []
        while len(pose_dict) > 0:
            print(f"len(pose_dict):{len(pose_dict)}")

            # 评估所有采样姿态
            values_sampled_poses = []
            keys = list(pose_dict.keys())
            for key in keys:
                pose = pose_dict[key]
                # 渲染当前姿态的全景图
                pano_rgb, pano_distance, pano_mask = self.render_pano(pose.cuda())
                # 计算视图完整度（未被遮挡的区域比例）
                view_completeness = torch.sum((1 - pano_mask * 1))/(pano_mask.shape[0] * pano_mask.shape[1])
                
                values_sampled_poses += [(key, view_completeness, pose)]
                torch.cuda.empty_cache() 
            if len(values_sampled_poses) < 1:
                break

            # 根据视图完整度排序，选择最不完整的视图进行修复
            values_sampled_poses = sorted(values_sampled_poses, key=lambda item: item[1])
            least_complete_view = values_sampled_poses[len(values_sampled_poses)*2//3]

            key, view_completeness, pose = least_complete_view
            print(f"least_complete_view:{view_completeness}")
            del pose_dict[key]

            # rendering rgb depth mask
            pano_rgb, pano_distance, pano_mask = self.render_pano(pose.cuda())

            # inpaint pano
            colors = pano_rgb.permute(1,2,0).clone()
            distances = pano_distance.unsqueeze(-1).clone()
            pano_inpaint_mask = pano_mask.clone()

            if pano_inpaint_mask.min().item() < .5:
                # inpainting pano
                colors, distances, normals = self.inpaint_new_panorama(idx=key, colors=colors, distances=distances, pano_mask=pano_inpaint_mask) # HWC, HWC, HW
                
                #apply_GeoCheck:
                perf_pose = pose.clone()
                perf_pose[0,3], perf_pose[1,3], perf_pose[2,3] = -pose[0,3], pose[2,3], 0 
                rays = gen_pano_rays(perf_pose, self.pano_height, self.pano_width)
                conflict_mask = self.sup_pool.geo_check(rays, distances.unsqueeze(-1))    # 0 conflict, 1 not conflict
                pano_inpaint_mask = pano_inpaint_mask * conflict_mask
                    
            # add new mesh
            self.pano_distance_to_mesh(colors.permute(2,0,1), distances, pano_inpaint_mask, pose=pose) #CHW, HW, HW

            # apply_GeoCheck:
            sup_mask = pano_inpaint_mask.clone()
            self.sup_pool.register_sup_info(pose=perf_pose, mask=sup_mask, rgb=colors, distance=distances.unsqueeze(-1), normal=normals)
            
            # save renderred
            panorama_tensor_pil = functions.tensor_to_pil(pano_rgb.unsqueeze(0))
            panorama_tensor_pil.save(f"{self.save_path}/renderred_pano_{key}.png")
            if self.save_details:
                depth_pil = Image.fromarray(colorize_single_channel_image(pano_distance.unsqueeze(0)/self.scene_depth_max))
                depth_pil.save(f"{self.save_path}/renderred_depth_{key}.png")        
                inpaint_mask_pil = Image.fromarray(pano_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")
                inpaint_mask_pil.save(f"{self.save_path}/mask_{key}.png")  
                inpaint_mask_pil = Image.fromarray(pano_inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")
                inpaint_mask_pil.save(f"{self.save_path}/inpaint_mask_{key}.png")  

            # save inpainted
            panorama_tensor_pil = functions.tensor_to_pil(colors.permute(2,0,1).unsqueeze(0))
            panorama_tensor_pil.save(f"{self.save_path}/inpainted_pano_{key}.png")
            depth_pil = Image.fromarray(colorize_single_channel_image(distances.unsqueeze(0)/self.scene_depth_max))
            depth_pil.save(f"{self.save_path}/inpainted_depth_{key}.png") 

            # collect pano images for GS training
            inpainted_panos_and_poses += [(colors.permute(2,0,1).unsqueeze(0), pose.clone())] #BCHW, 44
            
        return inpainted_panos_and_poses

    def inpaint_new_panorama(self, idx, colors, distances, pano_mask):
        """修复新的全景图
        Args:
            idx: 全景图索引
            colors: RGB颜色数据
            distances: 距离图数据
            pano_mask: 全景图遮罩
        Returns:
            inpainted_img: 修复后的图像
            inpainted_distances: 修复后的距离图
            inpainted_normals: 修复后的法线图
        """
        print(f"inpaint_new_panorama")

        # 必须先对遮罩进行膨胀处理
        mask = pano_mask.unsqueeze(-1)
        s_size = (9, 9)
        kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, s_size)
        kernel_s = torch.from_numpy(kernel_s).to(torch.float32).to(mask.device)
        mask = (mask[None, :, :, :] > 0.5).float()
        mask = mask.permute(0, 3, 1, 2)
        mask = dilation(mask, kernel=kernel_s)
        mask.permute(0, 2, 3, 1).contiguous().squeeze(0).squeeze(-1)

        distances = distances.squeeze()[..., None]
        mask = mask.squeeze()[..., None]

        inpainted_distances = None
        inpainted_normals = None

        # 使用修复器修复图像
        inpainted_img = self.inpainter.inpaint(idx, colors, mask)

        # 保持已渲染部分不变
        inpainted_img = colors * (1 - mask) + inpainted_img * mask
        inpainted_img = inpainted_img.cuda()

        # 使用几何预测器生成距离图和法线图
        inpainted_distances, inpainted_normals = self.geo_predictor(idx,
                                                                    inpainted_img,
                                                                    distances,
                                                                    mask=mask,
                                                                    reg_loss_weight=0.,
                                                                    normal_loss_weight=5e-2,
                                                                    normal_tv_loss_weight=5e-2)

        inpainted_distances = inpainted_distances.squeeze()
        return inpainted_img, inpainted_distances, inpainted_normals

    def load_pano(self):
        """加载全景图和深度图
        Returns:
            panorama_tensor: 全景图张量
            depth: 深度图数据
        """
        # 加载全景图像
        image_path = f"input/input_panorama.png"
        image = Image.open(image_path)
        if image.size[0] < image.size[1]:
            image = image.transpose(Image.TRANSPOSE)
        image = functions.resize_image_with_aspect_ratio(image, new_width=self.pano_width)
        panorama_tensor = torch.tensor(np.array(image))[...,:3].permute(2,0,1).unsqueeze(0).float()/255
        panorama_image_pil = functions.tensor_to_pil(panorama_tensor)

        # 深度缩放因子
        depth_scale_factor = 3.4092

        # 使用PanoFusion预测深度
        pano_fusion_distance_predictor = PanoFusionDistancePredictor()
        depth = pano_fusion_distance_predictor.predict(panorama_tensor.squeeze(0).permute(1,2,0)) #input:HW3
        depth = depth/depth.max() * depth_scale_factor
        print(f"pano_fusion_distance...[{depth.min(), depth.mean(),depth.max()}]")
        
        return panorama_tensor, depth  # panorama_tensor:BCHW, depth:HW

    def load_camera_poses(self, pano_center_offset=[0,0]):
        subset_path = f'input/Camera_Trajectory' # initial 6 poses are cubemaps poses
        files = os.listdir(subset_path)

        self.scene_depth_max = 4.0228885328450446

        pano_pose_44 = None
        pose_files = [f for f in files if f.startswith('camera_pose')]
        pose_files = sorted(pose_files)
        poses_name = pose_files
        poses = []
        
        # 遍历所有姿态文件
        for i, pose_name in enumerate(poses_name):
            with open(f'{subset_path}/{pose_name}', 'r') as f: 
                lines = f.readlines()
            pose_44 = []
            for line in lines:
                pose_44 += line.split()
            pose_44 = np.array(pose_44).reshape(4, 4).astype(float)
            if pano_pose_44 is None:
                pano_pose_44 = pose_44.copy()
                pano_pose_44_cubemaps = pose_44.copy()
                pano_pose_44[0,3] += pano_center_offset[0]
                pano_pose_44[2,3] += pano_center_offset[1]
            
            if i < 6:
                pose_relative_44 = pose_44 @ np.linalg.inv(pano_pose_44_cubemaps)  
            else:
                ### convert gt_pose to gt_relative_pose with pano_pose
                pose_relative_44 = pose_44 @ np.linalg.inv(pano_pose_44)

            pose_relative_44 = np.vstack((-pose_relative_44[0:1,:], -pose_relative_44[1:2,:], pose_relative_44[2:3,:], pose_relative_44[3:4,:]))
            pose_relative_44 = pose_relative_44 @ rot_z_world_to_cam(180).cpu().numpy()

            pose_relative_44[:3,3] *= self.pose_scale
            poses += [torch.tensor(pose_relative_44).float()] # w2c

        return pano_pose_44, poses

    def pano_to_perpective(self, pano_bchw, pitch, yaw, fov):
        rots = {
            'roll': 0.,
            'pitch': pitch,  # rotate vertical
            'yaw': yaw,  # rotate horizontal
        }

        perspective = equi2pers(
            equi=pano_bchw.squeeze(0),
            rots=rots,
            height=self.H,
            width=self.W,
            fov_x=fov,
            mode="bilinear",
        ).unsqueeze(0) # BCHW

        return perspective

    def pano_to_cubemap(self, pano_tensor, pano_depth_tensor=None): #BCHW, HW
        # 定义立方体贴图的6个面的方向（pitch和yaw角度）
        cubemaps_pitch_yaw = [(0, 0), (0, 3/2 * np.pi), (0, 1 * np.pi), (0, 1/2 * np.pi),\
                            (-1/2 * np.pi, 0), (1/2 * np.pi, 0)]
        pitch_yaw_list = cubemaps_pitch_yaw

        cubemaps = []
        cubemaps_depth = []
        # collect fov 90 cubemaps
        for view_idx, (pitch, yaw) in enumerate(pitch_yaw_list):
            view_rgb = self.pano_to_perpective(pano_tensor, pitch, yaw, 90)
            cubemaps += [view_rgb.cpu().clone()]
            if pano_depth_tensor is not None:
                view_depth = self.pano_to_perpective(pano_depth_tensor.unsqueeze(0).unsqueeze(0), pitch, yaw, 90)
                cubemaps_depth += [view_depth.cpu().clone()]
        return cubemaps, cubemaps_depth  # BCHW, BCHW

    def train_GS(self):
        if not self.scene:
            raise('Build 3D Scene First!')
        
        iterable_gauss = range(1, self.opt.iterations + 1)

        for iteration in iterable_gauss:
            self.gaussians.update_learning_rate(iteration)

            # Pick a random Camera
            viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam, mesh_pose = viewpoint_stack[iteration%len(viewpoint_stack)]

            # Render GS
            render_pkg = render(viewpoint_cam, self.gaussians, self.opt, self.background)
            render_image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
            
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(render_image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(render_image, gt_image))
            loss.backward()

            if self.save_details:
                if iteration % 200 == 0:
                    functions.write_image(f"{self.save_path}/Train_Ref_rgb_{iteration}.png", gt_image.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.)
                    functions.write_image(f"{self.save_path}/Train_GS_rgb_{iteration}.png", render_image.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.)

            with torch.no_grad():
                # Densification
                if iteration < self.opt.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                    
                    if (iteration % self.opt.opacity_reset_interval == 0 
                        or (self.opt.white_background and iteration == self.opt.densify_from_iter)
                    ):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

    def eval_GS(self, eval_GS_cams):
        viewpoint_stack = eval_GS_cams
        l1_val = 0
        ssim_val = 0
        psnr_val = 0
        framelist = []
        depthlist = []
        for i in range(len(viewpoint_stack)):
            viewpoint_cam, mesh_pose = viewpoint_stack[i]

            results = render(viewpoint_cam, self.gaussians, self.opt, self.background)
            frame, depth = results['render'], results['depth'].detach().cpu()

            framelist.append(
                np.round(frame.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
            depthlist.append(colorize_single_channel_image(depth.detach().cpu()/self.scene_depth_max))

        if self.save_details:
            for i, frame in enumerate(framelist):
                image = Image.fromarray(frame, mode="RGB")
                image.save(os.path.join(self.save_path, f"Eval_render_rgb_{i}.png"))
                functions.write_image(f"{self.save_path}/Eval_render_depth_{i}.png", depthlist[i])
        
        write_video(f"{self.save_path}/GS_render_video.mp4", framelist[6:], fps=30)
        write_video(f"{self.save_path}/GS_depth_video.mp4", depthlist[6:], fps=30)
        print("Result saved at: ", self.save_path)
            
    def run(self):
        # 设置默认张量类型
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # 加载相机姿态和全景图数据
        self.pano_pose, self.poses = self.load_camera_poses(self.pano_center_offset)
        pano_rgb, pano_depth = self.load_pano()
        panorama_tensor, init_depth = pano_rgb.squeeze(0).cuda(), pano_depth.cuda()

        # 处理深度边缘
        depth_edge = self.find_depth_edge(init_depth.cpu().detach().numpy(), dilate_iter=1)
        depth_edge_pil = Image.fromarray(depth_edge)
        depth_pil = Image.fromarray(visualize_depth_numpy(init_depth.cpu().detach().numpy())[0].astype(np.uint8))
        _, _ = save_rgbd(depth_pil, depth_edge_pil, f'depth_edge', 0, self.save_path)  
        depth_edge_inpaint_mask = ~(torch.from_numpy(depth_edge).cuda().bool()) 

        # 初始化监督信息池
        self.sup_pool = SupInfoPool()
        self.sup_pool.register_sup_info(pose=torch.eye(4).cuda(),
                                        mask=torch.ones([self.pano_height, self.pano_width]),
                                        rgb=panorama_tensor.permute(1,2,0),
                                        distance=init_depth.unsqueeze(-1))
        self.sup_pool.gen_occ_grid(256)

        # Pano2Mesh转换
        self.pano_distance_to_mesh(panorama_tensor, init_depth, depth_edge_inpaint_mask)

        # Mesh修复
        pose_dict = self.load_inpaint_poses()
        print(f"start inpainting with poses #{len(self.poses)}")
        inpainted_panos_and_poses = self.stage_inpaint_pano_greedy_search(pose_dict)

        # 直接导出mesh
        mesh_output_path = os.path.join(self.save_path, 'reconstructed_mesh.ply')
        self.save_mesh(mesh_output_path)
        print(f"Mesh saved at: {mesh_output_path}")

    def save_mesh(self, output_path):
        """保存mesh到PLY文件
        Args:
            output_path: 输出文件路径
        """
        # 准备顶点和颜色数据
        vertices = self.vertices.detach().cpu().numpy().T  # 转置为(N, 3)格式
        colors = (self.colors.detach().cpu().numpy().T * 255).astype(np.uint8)  # 转换为0-255范围
        faces = self.faces.detach().cpu().numpy().T  # 转置为(N, 3)格式

        # 创建PLY文件
        with open(output_path, 'w') as f:
            # 写入文件头
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # 写入顶点和颜色数据
            for vertex, color in zip(vertices, colors):
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")

            # 写入面片数据
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

pipeline = Pano2RoomPipeline()
pipeline.run()